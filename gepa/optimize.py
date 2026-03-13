"""GEPA prompt optimization script.

Optimizes a system prompt (or any text artifact) using GEPA's evolutionary
Pareto search with LLM reflection. The agent modifies this file to change:
  - TASK_LM: the model being optimized
  - REFLECTION_LM: the model doing reflection (should be stronger)
  - SEED: the initial prompt to optimize
  - METRIC: the evaluation function
  - TRAINSET / VALSET: the evaluation data
  - MAX_METRIC_CALLS: budget (number of evaluations)

Usage:
    export OPENAI_API_KEY=...  # or ANTHROPIC_API_KEY
    python optimize.py > run.log 2>&1

Grep-parsable output:
    val_score: 0.85
    best_prompt: <the optimized prompt>
"""

import json
import logging
import gepa

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION (agent modifies these)
# ============================================================

# Models
TASK_LM = "openai/gpt-4.1-nano"        # weaker model to give GEPA more room
REFLECTION_LM = "openai/gpt-5.4"      # flagship model for better reflection

# Budget
MAX_METRIC_CALLS = 100  # serious run

# Seed prompt to optimize
SEED = {
    "system_prompt": (
        "You are evaluating the quality of a code review comment. "
        "Read the code review below and classify it as either 'good' or 'bad'. "
        "A good review identifies a real, actionable problem and is technically correct. "
        "A bad review is vague, technically wrong, merely describes the code, or is a rubber stamp. "
        "Note: a review can be long and detailed but still bad if the advice is incorrect. "
        "A review can be short and still good if it pinpoints a real bug. "
        "Respond with exactly one word: good or bad"
    )
}

# ============================================================
# DATA (agent modifies these)
# ============================================================

def _d(input, answer):
    """Create a data instance with required fields."""
    return {"input": input, "answer": answer, "additional_context": {}}

# --- TRAINSET: 70 labeled code review examples (35 good, 35 bad) ---
# Includes borderline cases to break length/surface-level shortcuts
TRAINSET = [
    # ---- GOOD reviews (35) ----
    # Clear bug identification (kept from original)
    _d(
        "In `getUserProfile`, line 42: you're accessing `user.address.zipCode` without "
        "checking if `address` is null. If the user never set an address, this will throw "
        "a TypeError in production. Add an optional chain: `user.address?.zipCode`.",
        "good"
    ),
    _d(
        "Line 31: `for i in range(1, len(items))` skips the first item (index 0). The total "
        "will always be missing one element. Should be `range(len(items))` or `range(0, len(items))`.",
        "good"
    ),
    _d(
        "The `incrementCounter` method reads, increments, and writes back without any locking. "
        "Two threads hitting this concurrently will lose an increment. Either use an atomic "
        "operation like `INCR` in Redis or wrap this in a mutex.",
        "good"
    ),
    _d(
        "Line 55: `query = f\"SELECT * FROM users WHERE name = '{name}'\"` is a textbook SQL "
        "injection vulnerability. If `name` is `'; DROP TABLE users; --` the table is gone. "
        "Use parameterized queries: `cursor.execute('SELECT * FROM users WHERE name = %s', (name,))`.",
        "good"
    ),
    _d(
        "The `setInterval` in `usePolling` is never cleared when the component unmounts. Each "
        "mount adds another interval, and after navigating back and forth a few times you'll "
        "have dozens of timers running. Return a cleanup function from `useEffect`.",
        "good"
    ),
    _d(
        "The nested loop on lines 12-18 iterates over `users` x `permissions`, making this "
        "O(n*m). With 10k users and 500 permissions, that's 5M iterations per request. Build "
        "a set from `permissions` first for O(1) lookups, making it O(n+m).",
        "good"
    ),
    _d(
        "The JWT token is stored in `localStorage`, which is accessible to any XSS attack on "
        "the page. Consider using httpOnly cookies instead, which JavaScript cannot read. This "
        "is especially important since you're rendering user-generated HTML.",
        "good"
    ),
    _d(
        "In `isEligibleForDiscount`, the condition `age > 65 || memberSince < 2020` should "
        "use AND, not OR. Right now every member who joined before 2020 gets a senior discount "
        "regardless of age. Based on the spec, both conditions must be true.",
        "good"
    ),
    _d(
        "The catch block on line 45 is `except Exception: pass`. This silently swallows all "
        "errors including things like `KeyboardInterrupt` and `SystemExit`. At minimum, log "
        "the exception. Better: catch only the specific exception you expect (e.g., `ValueError`).",
        "good"
    ),
    _d(
        "Line 67: `log.info(f'Processing payment for user {user.email}, card {card_number}')` "
        "is logging full credit card numbers to the application log. This violates PCI-DSS. "
        "Mask the card number: `card_number[-4:].rjust(len(card_number), '*')`.",
        "good"
    ),
    # BORDERLINE GOOD: very concise but still actionable (breaks length heuristic)
    _d("Line 44: off-by-one. `<=` should be `<`, you'll read past the buffer.", "good"),
    _d("Null deref on line 12 when `user` is None. Add a guard.", "good"),
    _d("`open()` without `with` — file handle leaks on exception.", "good"),
    _d("Race condition: read-modify-write without lock on shared counter.", "good"),
    _d("SQL injection via string concat on line 88. Use parameterized query.", "good"),
    _d("Division by zero when `count` is 0 on line 23.", "good"),
    _d("`==` instead of `===` on line 15 — coerces `null` to `0` silently.", "good"),
    _d("Missing `await` on line 30 — promise result is discarded.", "good"),
    _d("Integer overflow: `a * b` can exceed MAX_SAFE_INTEGER here.", "good"),
    _d("Return value unchecked — `malloc` can return NULL.", "good"),
    # BORDERLINE GOOD: identifies real but subtle issues
    _d(
        "The `setTimeout(0)` on line 5 doesn't guarantee immediate execution — it goes to the "
        "back of the macrotask queue. If there are pending microtasks or I/O callbacks, this "
        "could fire much later than expected, causing the race condition in the test.",
        "good"
    ),
    _d(
        "This float comparison `if (total == 0.3)` will fail because `0.1 + 0.2` is "
        "0.30000000000000004 in IEEE 754. Use an epsilon: `Math.abs(total - 0.3) < 1e-9`.",
        "good"
    ),
    _d(
        "The `encodeURIComponent` call is missing on `searchQuery` before appending it to the "
        "URL. A query containing `&` or `=` will break the URL parsing and potentially expose "
        "an open redirect vulnerability.",
        "good"
    ),
    _d(
        "Using `new Date()` in the test makes it flaky — if the test runs at exactly midnight "
        "the date will be different from the expected value. Mock the clock or use a fixed date.",
        "good"
    ),
    _d(
        "The `for...in` loop on line 7 iterates over inherited prototype properties too, not "
        "just own properties. Use `Object.keys()` or `for...of` with `Object.entries()` instead.",
        "good"
    ),
    # BORDERLINE GOOD: good question that identifies a real concern
    _d(
        "What's the behavior when `retries` exceeds 5? I don't see a base case — this looks "
        "like it could recurse indefinitely and blow the stack.",
        "good"
    ),
    _d(
        "Is `createdAt` indexed? This query filters on it with 10M rows and I don't see an "
        "index in the migration files. Without one this is a full table scan on every request.",
        "good"
    ),
    _d(
        "The `Content-Type` isn't set on this response. Browsers will sniff the MIME type and "
        "could interpret JSON as HTML, opening an XSS vector if any user content is embedded.",
        "good"
    ),
    _d(
        "This regex allows `..` in the file path which means an attacker can traverse to "
        "`../../etc/passwd`. Normalize the path and verify it starts with the upload directory.",
        "good"
    ),
    _d(
        "Line 90: `str(price)` rounds the float, but you're storing money. With 10,000 "
        "transactions per day, rounding errors compound to real dollar amounts. Use Decimal.",
        "good"
    ),

    # ---- BAD reviews (35) ----
    # Obvious rubber stamps (kept)
    _d("LGTM", "bad"),
    _d("Looks good to me, ship it!", "bad"),
    _d("Approved. No comments.", "bad"),
    _d("+1, let's merge this.", "bad"),
    _d("Seems fine. Approving.", "bad"),
    # Vague (kept)
    _d("This could be better.", "bad"),
    _d("Maybe add some error handling?", "bad"),
    _d("Have you considered edge cases?", "bad"),
    _d("This doesn't feel right, but I can't put my finger on it.", "bad"),
    _d("Consider refactoring this part.", "bad"),
    # BORDERLINE BAD: long, detailed, specific-sounding BUT technically WRONG
    _d(
        "You should switch from `async/await` to raw `Promise.then()` chains here. "
        "Async/await has significant overhead because each `await` allocates a new microtask "
        "on the event loop, which adds ~50ms of latency per call. With 100 API calls, that's "
        "5 seconds of pure overhead. Promises avoid this because they resolve synchronously.",
        "bad"
    ),
    _d(
        "This `HashMap` usage is dangerous because Java's `HashMap.get()` has O(n) worst-case "
        "complexity when hash collisions occur. You should use a `TreeMap` instead, which "
        "guarantees O(log n) for all operations. The 2x memory overhead is worth the "
        "predictable performance.",
        "bad"
    ),
    _d(
        "The `Array.prototype.map()` on line 45 creates a new array in memory on every call. "
        "For a list of 50 items, that's 50 new objects being garbage collected per render. "
        "Replace this with a `for` loop that mutates the array in-place to avoid the allocation. "
        "This is a well-known React performance optimization.",
        "bad"
    ),
    _d(
        "I'd recommend wrapping every function in this module with a try/catch that logs the "
        "error and returns a default value. That way nothing ever throws unexpectedly and your "
        "error boundaries won't trigger. It's a best practice from the Erlang 'let it crash' "
        "philosophy adapted for JavaScript.",
        "bad"
    ),
    _d(
        "Since Python uses reference counting for garbage collection, circular references are "
        "never cleaned up. You should manually break the cycle by setting `parent.child = None` "
        "before the function returns. Otherwise, this will leak about 1KB per call, which adds "
        "up to gigabytes in production over a weekend.",
        "bad"
    ),
    # BORDERLINE BAD: specific about something completely trivial
    _d(
        "On line 23, the variable name `idx` should be `index` because three-letter abbreviations "
        "reduce readability. According to Clean Code chapter 2, meaningful names should be "
        "pronounceable. `idx` saves only 2 characters and isn't worth the cognitive overhead.",
        "bad"
    ),
    _d(
        "The import order here doesn't match the project convention. Standard library imports "
        "should come first, then third-party, then local. I count 3 imports out of order. "
        "This is documented in our style guide section 4.2.1.",
        "bad"
    ),
    _d(
        "Line 12 uses `const result = await fetch(url)` but our team convention is to use "
        "`const response = await fetch(url)`. We should be consistent — `result` implies the "
        "parsed data, while `response` correctly implies the raw HTTP response object.",
        "bad"
    ),
    _d(
        "I notice you're using tabs for indentation in this file but the rest of the project "
        "uses 2-space indentation. This will cause diff noise when someone else edits the file. "
        "Please run prettier with the project config before merging.",
        "bad"
    ),
    _d(
        "The spacing around the `=` operator on lines 15, 22, and 37 is inconsistent. Lines 15 "
        "and 37 have spaces (`x = 5`) but line 22 doesn't (`x=5`). This inconsistency makes "
        "the code harder to scan visually. Please standardize.",
        "bad"
    ),
    # BORDERLINE BAD: confidently wrong fix suggestion
    _d(
        "The `useEffect` dependency array is missing `count`. React's exhaustive-deps rule "
        "requires all variables used inside the effect to be listed. Add `count` to the array "
        "to fix the stale closure. Yes, this means the effect will re-run on every count change, "
        "but that's the correct React pattern.",
        "bad"
    ),
    _d(
        "You should cache this database query result in a global variable so it's only fetched "
        "once when the server starts. Database connections are expensive and this endpoint is "
        "called frequently. Storing the result globally ensures maximum performance with zero "
        "latency on subsequent calls.",
        "bad"
    ),
    # BORDERLINE BAD: describes the code without adding value (but sounds thorough)
    _d(
        "Looking at the control flow: the function first validates the input on line 5, then "
        "transforms it on line 12 using the helper function defined above, catches any errors "
        "in the try/catch block on lines 15-20, and finally returns the result on line 22. "
        "The error handling follows the standard pattern we use elsewhere in the codebase.",
        "bad"
    ),
    _d(
        "I traced through the logic carefully. When `status` is 'active', we enter the first "
        "branch which calls `processActive()`. When it's 'pending', we hit the else-if and "
        "call `processPending()`. The default case calls `processUnknown()`. This matches the "
        "state machine diagram in the design doc.",
        "bad"
    ),
    # BORDERLINE BAD: identifies a "problem" that isn't one
    _d(
        "The `finally` block on line 30 will execute even if the function returns early on "
        "line 25. This could cause unexpected behavior if the cleanup code in `finally` depends "
        "on state that was never initialized. I'd remove the `finally` and put cleanup at each "
        "exit point instead.",
        "bad"
    ),
    _d(
        "Using `Object.freeze()` on the config object means you can't add new properties later "
        "if requirements change. This makes the code inflexible. I'd remove the freeze and use "
        "a comment like `// don't modify` instead, which communicates intent without restricting "
        "future changes.",
        "bad"
    ),
    # BORDERLINE BAD: good diagnostic but actively harmful suggestion
    _d(
        "The password validation on line 18 rejects passwords with spaces. Some users put spaces "
        "in their passwords for memorability. You should `trim()` the password before validating "
        "and storing it, so spaces at the beginning and end are removed but internal spaces are "
        "kept. This is a common UX improvement.",
        "bad"
    ),
    # BORDERLINE BAD: long compliment disguised as review
    _d(
        "This is really well-architected. The separation of concerns between the service layer "
        "and the repository layer is textbook clean architecture. I especially like how you've "
        "used dependency injection for the database client — it makes testing much easier. The "
        "error handling is thorough and the naming conventions are consistent throughout. Great "
        "work on this refactor, it's significantly better than what was here before.",
        "bad"
    ),
    _d(
        "I've reviewed the entire PR and I want to commend the thoroughness. Every function "
        "has proper JSDoc comments, the test coverage looks comprehensive, and the commit "
        "history is clean and well-organized. The migration is backwards-compatible which shows "
        "good engineering judgment. No concerns from my side — this is production-ready code.",
        "bad"
    ),
    _d(
        "I trust your judgment on this one. You've been working on this module for a while "
        "and understand the constraints better than anyone on the team. The approach looks "
        "reasonable and aligns with what we discussed in last week's design review. Approving.",
        "bad"
    ),
]

# --- VALSET: 30 labeled code review examples (15 good, 15 bad) ---
# Heavy on borderline cases to test rubric discriminating power
VALSET = [
    # ---- GOOD reviews (15) ----
    # Clear good (8)
    _d(
        "In `sendEmail`, the `to` parameter is passed directly to the SMTP library without "
        "any validation. An attacker could inject headers with `\\r\\nBCC: spam@evil.com` and "
        "use your server as an open relay. Sanitize the input or use a library that handles this.",
        "good"
    ),
    _d(
        "The `mergeConfigs` function uses the spread operator `{...defaults, ...userConfig}` "
        "but this is a shallow merge. Nested objects like `userConfig.database.pool` will "
        "completely overwrite `defaults.database`, losing `defaults.database.host`. Use a "
        "deep merge utility.",
        "good"
    ),
    _d(
        "In the `handleUpload` endpoint, `req.file.size` is checked after the entire file is "
        "already in memory. A 2GB upload will OOM the server before validation runs. Set "
        "`limits: { fileSize: 10 * 1024 * 1024 }` in the multer config to reject early.",
        "good"
    ),
    _d(
        "The `comparePasswords` function uses `===` for string comparison, which is vulnerable "
        "to timing attacks. Use `crypto.timingSafeEqual(Buffer.from(a), Buffer.from(b))` to "
        "prevent an attacker from guessing the hash character by character.",
        "good"
    ),
    _d(
        "The `hashPassword` function uses MD5, which is cryptographically broken and can be "
        "brute-forced in seconds with modern GPUs. Use bcrypt or argon2 with a proper salt "
        "and work factor of at least 10.",
        "good"
    ),
    _d(
        "The `calculateShipping` function returns a float like 9.999999999 due to floating-point "
        "arithmetic. For money calculations, use `Decimal` (Python) or integer cents to avoid "
        "rounding errors that compound across thousands of orders.",
        "good"
    ),
    _d(
        "Line 8: `del items[i]` inside a `for i in range(len(items))` loop modifies the list "
        "while iterating, which will skip elements and eventually raise IndexError. Build a "
        "new list with a comprehension: `items = [x for x in items if not should_remove(x)]`.",
        "good"
    ),
    _d(
        "The `convertCurrency` function caches exchange rates forever after first fetch. "
        "Rates change constantly — a stale rate could cause users to be charged the wrong "
        "amount. Add a TTL of 15 minutes and re-fetch when expired.",
        "good"
    ),
    # Borderline good: very short (7)
    _d("Double-free on line 45. `free(ptr)` called again in the error path.", "good"),
    _d("Unbounded recursion when `depth` param is negative. Add base case.", "good"),
    _d("TOCTOU on the file check — race between `exists()` and `open()`.", "good"),
    _d("Signed integer used for size on line 8. Negative length = buffer overread.", "good"),
    _d("`strncat` third arg should be `sizeof(buf) - strlen(buf) - 1`, not `sizeof(buf)`.", "good"),
    _d("Missing `break` in switch case at line 70 — falls through to default.", "good"),
    _d("Infinite loop: `while (i < len)` but `i` is never incremented inside the body.", "good"),

    # ---- BAD reviews (15) ----
    # Obvious bad (5)
    _d("Looks great, no issues!", "bad"),
    _d("Ship it!", "bad"),
    _d("Nit: extra whitespace on line 33.", "bad"),
    _d("Did you run the tests?", "bad"),
    _d("Could you add more tests?", "bad"),
    # Borderline bad: long, detailed, specific BUT wrong or harmful (10)
    _d(
        "You should always use `parseInt(x)` instead of `Number(x)` for string-to-number "
        "conversion because `Number()` is slower — it has to handle floats, scientific notation, "
        "and special values like Infinity. `parseInt` is a simpler operation that only handles "
        "integers, so the V8 engine can optimize it much better. In benchmarks, `parseInt` is "
        "3-5x faster across all common inputs.",
        "bad"
    ),
    _d(
        "I'd suggest replacing all `const` declarations with `let` in this file. `const` only "
        "prevents reassignment, not mutation, so it gives a false sense of immutability. Using "
        "`let` everywhere is more honest about what JavaScript actually guarantees and avoids "
        "confusing junior developers who think `const` means the value can't change.",
        "bad"
    ),
    _d(
        "The `===` comparison on line 34 should be `==` because strict equality doesn't do type "
        "coercion, which means comparing a string `'5'` with a number `5` will fail. Since the "
        "API can return either type depending on the endpoint, `==` is actually safer and more "
        "robust here.",
        "bad"
    ),
    _d(
        "You're catching the error and re-throwing it on line 55, which seems redundant. I'd "
        "remove the try/catch entirely so exceptions propagate naturally. The calling code should "
        "be responsible for error handling, not this utility function. Each function should do "
        "one thing, and error handling is a separate concern.",
        "bad"
    ),
    _d(
        "This `Map` could be replaced with a plain object `{}` for simplicity. Maps add "
        "unnecessary complexity — they have a different iteration API (`entries()` vs `for...in`), "
        "they're not JSON-serializable, and they use more memory. A plain object does everything "
        "a Map does and is more idiomatic JavaScript.",
        "bad"
    ),
    _d(
        "Great attention to detail in this PR. I walked through every line and the logic is "
        "sound. The database queries are well-structured, the error messages are user-friendly, "
        "and the input validation covers all the edge cases I can think of. This is a model PR "
        "for the team to follow. Approved with enthusiasm.",
        "bad"
    ),
    _d(
        "I traced the execution path from the controller through the service layer into the "
        "repository. The controller validates input and passes it to the service, which applies "
        "business logic and calls the repository, which executes the SQL query. The response "
        "is built in the controller from the service result. Standard MVC flow, no issues.",
        "bad"
    ),
    _d(
        "Hmm, I think using `reduce` here instead of `map` + `filter` would be more elegant. "
        "With `reduce`, you can combine both operations into a single pass over the array, "
        "which is both more performant and more functional in style. The FP purists on the "
        "team would appreciate the cleaner abstraction.",
        "bad"
    ),
    _d(
        "I notice the `finally` block on line 30 always runs cleanup code. While technically "
        "correct, this pattern can be confusing for developers unfamiliar with `try/catch/finally` "
        "semantics. I'd suggest refactoring to explicit cleanup calls at each return point for "
        "clarity, even though it means duplicating the cleanup code in 3 places.",
        "bad"
    ),
    _d(
        "The `try/finally` here should be replaced with `try/catch/finally`. A `try` block "
        "without a `catch` is incomplete and will cause unhandled exceptions to crash the "
        "process. Always include a `catch` block, even if it just re-throws, to ensure proper "
        "error handling chain. This is a common anti-pattern in Node.js.",
        "bad"
    ),
]

# ============================================================
# RUN
# ============================================================

def main():
    log.info("Starting GEPA optimization")
    log.info(f"Task LM: {TASK_LM}")
    log.info(f"Reflection LM: {REFLECTION_LM}")
    log.info(f"Budget: {MAX_METRIC_CALLS} metric calls")
    log.info(f"Train: {len(TRAINSET)} examples, Val: {len(VALSET)} examples")
    log.info(f"Seed prompt: {SEED['system_prompt'][:100]}...")

    result = gepa.optimize(
        seed_candidate=SEED,
        trainset=TRAINSET,
        valset=VALSET,
        task_lm=TASK_LM,
        reflection_lm=REFLECTION_LM,
        max_metric_calls=MAX_METRIC_CALLS,
        display_progress_bar=True,
    )

    # Extract results
    best = result.best_candidate
    best_idx = result.best_idx
    val_score = result.val_aggregate_scores[best_idx]

    log.info("=" * 60)
    log.info("RESULTS")
    log.info("=" * 60)

    # Grep-parsable output
    print(f"\nval_score: {val_score:.6f}")
    print(f"best_prompt: {json.dumps(best)}")

    # Human-readable
    log.info(f"Val score: {val_score}")
    log.info(f"Best prompt: {best.get('system_prompt', str(best))}")
    log.info(f"Candidates explored: {len(result.candidates)}")
    log.info(f"Total metric calls: {result.total_metric_calls}")
    log.info("Optimization completed.")


if __name__ == "__main__":
    main()
