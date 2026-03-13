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
TASK_LM = "openai/gpt-5-mini"           # latest mini model per program.md
REFLECTION_LM = "openai/gpt-5.4"      # flagship model for better reflection

# Budget
MAX_METRIC_CALLS = 500  # stage 2: iterative seed from stage 1

# Seed prompt to optimize
SEED = {
    "system_prompt": (
        "You are performing a strict binary classification task on exactly one code review comment.\n\n"
        "Output exactly one word: `good` or `bad`. Nothing else.\n\n"
        "## Core standard\n"
        "Label `good` only if ALL of these are satisfied:\n"
        "1. Specific: identifies a concrete issue in the code.\n"
        "2. Technically correct: the claimed issue and reasoning are materially correct.\n"
        "3. Actionable: suggests or implies an appropriate fix.\n"
        "4. Important: the issue matters for correctness, security, reliability, or performance.\n"
        "5. Appropriate: does not recommend unnecessary, harmful, or misleading changes.\n\n"
        "If any check fails, output `bad`.\n\n"
        "## What should be `bad`\n"
        "- praise, approval, conversational commentary\n"
        "- vague or generic advice\n"
        "- style-only, formatting, naming, idioms, conventions\n"
        "- process or tooling complaints\n"
        "- speculative, exaggerated, or absolutist claims\n"
        "- technically incorrect or misleading reasoning\n"
        "- recommending unnecessary or harmful changes\n"
        "- pedantic observations where the practical impact is negligible or near-zero\n"
        "- technically correct observations about issues that have no real-world consequence\n\n"
        "## Conservative policy\n"
        "- Prefer `bad` when uncertain.\n"
        "- Do not reward confidence, detail, or length alone.\n"
        "- A short comment can be `good` if it identifies a real bug correctly.\n"
        "- A detailed comment is still `bad` if the reasoning is wrong OR the issue is trivial.\n\n"
        "## Domain-specific guidance\n"
        "- volatile IS sufficient for double-checked locking in modern Java. Claiming otherwise is `bad`.\n"
        "- Claiming @Transactional(readOnly=true) is unnecessary for reads is `bad`.\n"
        "- Absolutist claims like 'recursion is never safe in Java' are `bad`.\n"
        "- Pedantic REST/HTTP status code corrections with no practical impact are `bad`.\n"
        "- Observations about negligible statistical bias (e.g. 1 in 2^53) are `bad`.\n\n"
        "Return exactly one word: good or bad"
    )
}

# ============================================================
# DATA (agent modifies these)
# ============================================================

def _d(input, answer):
    """Create a data instance with required fields."""
    return {"input": input, "answer": answer, "additional_context": {}}

# --- TRAINSET: 100 labeled code review examples (50 good, 50 bad) ---
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

    # ---- MORE TRAINING: ultra-hard (10 good, 10 bad) ----
    # GOOD: framework/language-specific gotchas
    _d(
        "The `@PostConstruct` method calls `this.fetchConfig()` which is `@Async`. But "
        "`@Async` is proxy-based — calling it from within the same class bypasses the proxy. "
        "The config fetch runs synchronously, blocking startup for 30 seconds.",
        "good"
    ),
    _d(
        "The `WeakHashMap` is keyed by `String`. String literals are interned by the JVM and "
        "never garbage collected, so entries with literal keys will never be evicted. This "
        "effectively makes it a regular HashMap for most of your keys, defeating the purpose.",
        "good"
    ),
    _d(
        "`Arrays.asList()` returns a fixed-size list backed by the array. The `add()` call on "
        "line 20 will throw `UnsupportedOperationException` at runtime. Use `new ArrayList<>(Arrays.asList(...))` "
        "if you need a mutable list.",
        "good"
    ),
    _d(
        "The `SimpleDateFormat` on line 5 is a class field, but `SimpleDateFormat` is not "
        "thread-safe. Concurrent requests will corrupt the internal calendar state, producing "
        "wrong dates intermittently. Use `DateTimeFormatter` (immutable) or ThreadLocal.",
        "good"
    ),
    _d(
        "The `PreparedStatement` is created inside the loop but never closed. Each iteration "
        "leaks a statement. After ~100 iterations you'll hit the database cursor limit and "
        "all subsequent queries fail. Move the creation outside the loop and reuse it.",
        "good"
    ),
    _d("Dangling pointer: `buf` freed on line 12 but returned on line 15.", "good"),
    _d("Uninitialized `sum` variable — will contain garbage on first use.", "good"),
    _d("Buffer overflow: `sprintf` with `%s` and no length limit on user input.", "good"),
    _d("`strcmp` returns 0 for equal, not true. The `if (strcmp(...))` is inverted.", "good"),
    _d("Use-after-move: `vec` accessed on line 20 after `std::move(vec)` on line 18.", "good"),

    # BAD: wrong-but-confident framework claims
    _d(
        "React's `useCallback` should wrap every callback function in your component. Without "
        "it, every render creates a new function reference which causes all child components "
        "to re-render. This is the #1 performance issue in React apps. Wrap all 12 callbacks "
        "in this component with `useCallback` for a significant speedup.",
        "bad"
    ),
    _d(
        "The `@Transactional(readOnly = true)` annotation on this read method is unnecessary "
        "overhead. Read operations don't need transactions because they can't corrupt data. "
        "Removing it will reduce connection pool usage and improve throughput by ~15% based "
        "on our load tests last quarter.",
        "bad"
    ),
    _d(
        "Instead of catching `NumberFormatException` here, use a regex to validate the string "
        "is numeric before parsing. Regex validation is always faster than exception handling "
        "because exceptions require stack unwinding which is O(n) in call depth. The regex "
        "`^\\d+$` covers all valid integer inputs.",
        "bad"
    ),
    _d(
        "The `ConcurrentHashMap` here could be replaced with a regular `HashMap` wrapped in "
        "`Collections.synchronizedMap()`. They have identical thread-safety guarantees but "
        "`synchronizedMap` is simpler and has lower memory overhead because it doesn't need "
        "the segment array that `ConcurrentHashMap` uses internally.",
        "bad"
    ),
    _d(
        "This `CompletableFuture.supplyAsync()` call doesn't specify an executor, so it uses "
        "the common ForkJoinPool which has exactly Runtime.getRuntime().availableProcessors() "
        "threads. But since this is an I/O operation, it will block a compute thread. Pass a "
        "custom `Executors.newCachedThreadPool()` to avoid this. Cached thread pools create "
        "unlimited threads on demand with zero overhead.",
        "bad"
    ),
    # BAD: identifies issue + wrong fix
    _d(
        "The date parsing on line 8 will fail for dates like '2024-02-29' (leap year) because "
        "you're using `LocalDate.parse` with a strict resolver. The fix is to use "
        "`ResolverStyle.LENIENT` which automatically adjusts invalid dates (e.g., Feb 29 in a "
        "non-leap year becomes Mar 1). This is the standard pattern for robust date handling.",
        "bad"
    ),
    _d(
        "Good catch on the null check, but instead of throwing `NullPointerException`, I'd "
        "use `Optional.ofNullable(value).orElseThrow()`. This is more idiomatic Java and the "
        "stack trace is cleaner because `Optional.orElseThrow()` throws `NoSuchElementException` "
        "which is more descriptive than NPE.",
        "bad"
    ),
    _d(
        "The `synchronized(this)` block is a performance bottleneck. Replace it with a "
        "`ReentrantLock` for better throughput. `ReentrantLock` is always faster than "
        "`synchronized` because it uses CAS operations instead of OS-level mutex, and it "
        "supports fairness which prevents thread starvation.",
        "bad"
    ),
    _d(
        "This recursion depth could be a problem. Java's default stack size is only 512KB "
        "which limits recursion to about 500 frames. Increase the stack size to 8MB with "
        "`-Xss8m` JVM flag. This is a common fix and has no downside since modern systems "
        "have plenty of RAM.",
        "bad"
    ),
    _d(
        "The `equals()` method should use `instanceof` instead of `getClass()` for the type "
        "check. `getClass()` breaks the Liskov Substitution Principle because subclasses won't "
        "be considered equal to their parent class. Using `instanceof` is more flexible and is "
        "the recommended approach in Effective Java.",
        "bad"
    ),

    # ---- TRAINING: "sounds right but is wrong" pattern (5 good, 5 bad) ----
    # GOOD: genuinely correct concerns about JVM/language internals
    _d(
        "The `String.intern()` call on every request adds to the JVM string pool which is "
        "never garbage collected in older JVMs. With 1M unique strings per day, you'll slowly "
        "leak memory in the PermGen space (Java 7) or Metaspace (Java 8+).",
        "good"
    ),
    _d(
        "The `ConcurrentHashMap.size()` call inside the if-check is not atomic with the "
        "subsequent `put()`. Between the size check and the put, another thread could have "
        "added an element, exceeding your limit. Use `compute()` or `computeIfAbsent()` for "
        "atomic check-and-act.",
        "good"
    ),
    _d(
        "The `AtomicInteger.get()` followed by `compareAndSet()` on the next line is a "
        "classic ABA problem. Another thread could change the value from A to B to A between "
        "the get and the CAS, making your CAS succeed when it shouldn't.",
        "good"
    ),
    _d(
        "The `finalize()` method is being used for resource cleanup but finalization is "
        "deprecated since Java 9 and unreliable — the JVM makes no guarantee about when or "
        "if finalizers run. Use `try-with-resources` or `Cleaner` instead.",
        "good"
    ),
    _d(
        "Autoboxing `int` to `Integer` inside this tight loop allocates ~10M objects per "
        "second. Use `IntStream` or primitive `int[]` instead of `List<Integer>` to avoid "
        "the GC pressure that's causing the latency spikes in production.",
        "good"
    ),
    # BAD: sounds technically right but is wrong or contextless
    _d(
        "The `volatile` keyword isn't enough for thread safety on the `singleton` field. "
        "You need `synchronized` because `volatile` only provides visibility, not atomicity. "
        "Without `synchronized`, the double-checked locking pattern is broken because the JVM "
        "can reorder the object initialization and field assignment.",
        "bad"
    ),
    _d(
        "This tail-recursive function should be rewritten iteratively because Java, unlike "
        "Scala, never performs tail-call optimization. There is literally no way to make "
        "recursion safe in Java — every recursive call adds a stack frame, and the only "
        "solution is to rewrite as a loop. No exceptions.",
        "bad"
    ),
    _d(
        "The HTTP 200 response code is technically wrong for this DELETE endpoint. According "
        "to RFC 7231 section 6.3.1, 200 should only be used when the response includes a "
        "representation of the action's result. Since this returns an empty body, you must "
        "use 204 No Content. Using the wrong status code will break well-behaved REST clients.",
        "bad"
    ),
    _d(
        "The `LocalDate.parse('2024-02-29')` call will fail because Java's default strict "
        "parsing mode rejects leap year dates. You need to use `ResolverStyle.SMART` to "
        "handle leap years correctly. This is a common gotcha that causes production "
        "failures every February 29th.",
        "bad"
    ),
    _d(
        "This code creates a new `SimpleDateFormat` on every call which is wasteful. Since "
        "`SimpleDateFormat` is thread-safe, you should create a single static instance and "
        "reuse it across all threads. This will reduce object allocation by 99% and "
        "significantly improve GC behavior.",
        "bad"
    ),
]

# --- VALSET: 70 labeled code review examples (35 good, 35 bad) ---
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

    # ---- ADVERSARIAL: extremely subtle borderline cases (20 more: 10 good, 10 bad) ----

    # GOOD: correct analysis, terse/unusual style
    _d(
        "This will deadlock. `mutex.lock()` on line 10, then `other_mutex.lock()` on line 15. "
        "But `processB` acquires them in the opposite order. Classic ABBA deadlock.",
        "good"
    ),
    _d(
        "The `toString()` override allocates a new StringBuilder on every call. This is on the "
        "hot path (called once per log line, ~50k/sec). Pre-allocate or cache the result.",
        "good"
    ),
    _d(
        "The comparison `a == b` where both are `float` is unreliable for equality. Use "
        "`abs(a - b) < epsilon` or `math.isclose(a, b)`.",
        "good"
    ),
    _d(
        "Storing user passwords in the session object means they persist in memory and could "
        "be leaked in a heap dump. Clear the password field after authentication.",
        "good"
    ),
    _d(
        "The `catch` on line 22 catches `Throwable` which includes `OutOfMemoryError` and "
        "`StackOverflowError`. Catching these will mask fatal JVM issues. Catch `Exception` "
        "instead.",
        "good"
    ),
    # GOOD: sounds like a complaint but identifies a real issue
    _d(
        "Why are we re-implementing Base64 encoding by hand? There are at least 3 bugs in this "
        "implementation: it doesn't handle padding correctly, the index table is wrong for "
        "characters 62-63, and it doesn't work with Unicode input. Use `btoa()` or `Buffer.from`.",
        "good"
    ),
    _d(
        "This entire retry mechanism is broken. The `catch` block sets `shouldRetry = true` but "
        "the `while` condition checks `retryCount < MAX_RETRIES` — and `retryCount` is never "
        "incremented. This will retry forever on any failure.",
        "good"
    ),
    # GOOD: identifies issue through a question (but the question IS the issue)
    _d(
        "What happens when two users submit the same coupon code at the same time? I don't see "
        "any locking on the `used_count` column. Both transactions will read the same count, "
        "both will allow the coupon, and you'll exceed the usage limit.",
        "good"
    ),
    _d(
        "Is this intentional? The `fallthrough` on line 45 means case 'admin' also executes "
        "case 'user' logic. An admin would get both permission sets applied, which contradicts "
        "the permission model described in the ticket.",
        "good"
    ),
    _d(
        "Line 33: `response.headers['X-Request-Id']` — the header name is case-sensitive in "
        "HTTP/2 and must be lowercase. This will return undefined in HTTP/2 environments. Use "
        "`response.headers['x-request-id']`.",
        "good"
    ),

    # BAD: sounds technical and specific but the recommendation is harmful
    _d(
        "The `immutable` flag on this Redux reducer state is unnecessary. JavaScript objects "
        "are passed by reference, so mutations are cheap and efficient. Remove the immutability "
        "library and mutate state directly in the reducer — this will simplify the code "
        "significantly and improve performance by avoiding unnecessary object copies.",
        "bad"
    ),
    _d(
        "Instead of this async generator, you should load the entire 500MB CSV file into memory "
        "first, then process it. Reading in chunks adds complexity and the OS page cache will "
        "handle memory management for you. Simpler code is always better than premature "
        "optimization for memory.",
        "bad"
    ),
    # BAD: technically correct observation but not actionable / not a real issue
    _d(
        "I notice this function has a cyclomatic complexity of 12. According to the McCabe "
        "threshold, anything above 10 should be refactored. However, I can see that each "
        "branch handles a distinct case, so maybe it's fine. Just wanted to flag it for "
        "awareness.",
        "bad"
    ),
    _d(
        "This code uses `let` inside a `for` loop, which creates a new binding per iteration. "
        "In older JavaScript engines (pre-V8 6.0), this was slower than `var`. It's fine in "
        "modern engines, but I thought I'd mention it for historical context.",
        "bad"
    ),
    # BAD: mixture of correct and incorrect — net negative
    _d(
        "Two issues: (1) The SQL query on line 30 should use `COALESCE` instead of `IFNULL` "
        "for cross-database compatibility — good point. (2) You should also wrap the entire "
        "endpoint in a `synchronized` block to prevent race conditions, since HTTP requests "
        "can arrive concurrently and corrupt the response object.",
        "bad"
    ),
    # BAD: armchair architecture without identifying a problem
    _d(
        "Looking at the overall design, I think this would benefit from an event-driven "
        "architecture with a message queue. The current synchronous request/response pattern "
        "works but doesn't scale. We should consider RabbitMQ or Kafka for decoupling the "
        "producer and consumer sides. This would also make it easier to add new consumers "
        "in the future without modifying the producer.",
        "bad"
    ),
    _d(
        "Have you considered using the Strategy pattern here? The switch statement could be "
        "replaced with a polymorphic dispatch table, where each case is a separate class "
        "implementing a common interface. This would make the code more extensible and follow "
        "the Open/Closed Principle from SOLID.",
        "bad"
    ),
    # BAD: claims to have tested something but gives wrong conclusion
    _d(
        "I benchmarked `for...of` vs `forEach` on an array of 10,000 elements and `for...of` "
        "was consistently 2x slower. The overhead comes from the iterator protocol — calling "
        "`.next()` on each iteration creates a new object. You should replace the `for...of` "
        "with `forEach` for better performance in this hot loop.",
        "bad"
    ),
    # BAD: meta-commentary about the PR process, not the code
    _d(
        "This PR is quite large — 47 files changed. It would have been easier to review if "
        "it were broken into smaller PRs, one per feature. For future reference, try to keep "
        "PRs under 200 lines of diff. That said, I've reviewed everything and have no "
        "blocking concerns with the actual code changes.",
        "bad"
    ),
    # BAD: correct concern but recommends doing nothing
    _d(
        "Technically this has a race condition between the read on line 5 and the write on "
        "line 12. However, in practice this endpoint handles maybe 10 requests per minute "
        "so the window is tiny. I wouldn't bother fixing it — the locking overhead would "
        "outweigh the practically nonexistent risk. Leaving as-is is fine.",
        "bad"
    ),

    # ---- ULTRA-HARD: context-dependent and judgment-heavy (20 more: 10 good, 10 bad) ----

    # GOOD: identifies a real bug through careful reasoning
    _d(
        "The `hashCode()` override returns `name.hashCode()` but `equals()` compares both "
        "`name` AND `age`. Two objects with the same name but different ages will hash to the "
        "same bucket but not be equal — this violates the hashCode contract and will cause "
        "silent data loss in HashMaps.",
        "good"
    ),
    _d(
        "The `Comparator` on line 8 does `return a.score - b.score` which overflows for large "
        "negative values. With `a.score = Integer.MIN_VALUE` and `b.score = 1`, the subtraction "
        "wraps to a positive number, reversing the sort order. Use `Integer.compare(a.score, b.score)`.",
        "good"
    ),
    _d(
        "The `@Transactional` annotation is on a private method. Spring's proxy-based AOP "
        "cannot intercept private methods, so this transaction boundary is silently ignored. "
        "Move the annotation to a public method or use AspectJ weaving.",
        "good"
    ),
    _d(
        "The CORS config allows `Access-Control-Allow-Origin: *` with `credentials: true`. "
        "Browsers reject this combination — you can't use wildcard origin with credentials. "
        "Either specify the exact allowed origins or remove credential support.",
        "good"
    ),
    _d(
        "The `Iterator.remove()` call inside `stream().forEach()` will throw "
        "`ConcurrentModificationException`. Streams don't support structural modification of the "
        "source during traversal. Use `Collection.removeIf()` instead.",
        "good"
    ),
    _d(
        "The regex `.*` in the route definition matches greedily across path segments. "
        "A request to `/api/users/../admin/config` will match this route and bypass the auth "
        "middleware. Use `[^/]*` to restrict matching to a single segment.",
        "good"
    ),
    _d(
        "`Double.NaN == Double.NaN` returns false in Java. The score filter on line 40 "
        "silently drops all NaN scores instead of handling them. Use `Double.isNaN()` for the check.",
        "good"
    ),
    _d(
        "The enum `valueOf()` on line 15 throws `IllegalArgumentException` for unknown values. "
        "Since the input comes from user-submitted JSON, any typo will crash the endpoint with a 500. "
        "Use a lookup map with a default or catch the exception.",
        "good"
    ),
    # GOOD: identifies a subtle correctness issue in test code
    _d(
        "This test mocks `clock.now()` to return a fixed time, but the production code calls "
        "`Instant.now()` directly (not through the clock). The test passes by coincidence — "
        "the mock is never invoked. Inject the `Clock` as a dependency for this to actually test "
        "the time-dependent behavior.",
        "good"
    ),
    _d(
        "The `assertThat(result).contains('success')` assertion is checking the toString() of "
        "the Response object, not the body. It passes because toString() includes the status, "
        "which contains the string 'success'. Check `result.getBody()` instead.",
        "good"
    ),

    # BAD: sounds like it identifies a bug but the analysis is wrong
    _d(
        "The `volatile` keyword on the `initialized` flag isn't sufficient for thread safety "
        "here. You need a full `synchronized` block around the double-checked locking pattern "
        "because `volatile` only guarantees visibility, not atomicity. Without `synchronized`, "
        "two threads could both see `initialized = false` and initialize twice.",
        "bad"
    ),
    _d(
        "This recursive function has a stack depth of O(n) which means for n=100,000 you'll "
        "get a StackOverflowError. But the real issue is that Java doesn't support tail-call "
        "optimization, so even tail-recursive code will overflow. You need to rewrite this "
        "as an iterative loop — there's no other option in Java.",
        "bad"
    ),
    # BAD: correct observation wrapped in wrong conclusion
    _d(
        "The `StringBuilder` is being shared across threads without synchronization. However, "
        "the fix isn't to add synchronization — that would be too slow. Instead, switch to "
        "`StringBuffer`, which is the thread-safe version of `StringBuilder` and is designed "
        "exactly for this use case. `StringBuffer` has no performance overhead compared to "
        "`StringBuilder` because modern JVMs optimize the synchronization away.",
        "bad"
    ),
    # BAD: identifies a theoretical problem that can't happen in this context
    _d(
        "The `ArrayList` isn't thread-safe. If another thread adds elements while this loop "
        "iterates, you'll get a `ConcurrentModificationException`. Consider using "
        "`CopyOnWriteArrayList` to avoid this. The overhead is minimal for small lists.",
        "bad"
    ),
    # BAD: confidently recommends an anti-pattern
    _d(
        "Rather than throwing exceptions from this validation method, I'd recommend returning "
        "null to indicate failure. The caller can then check for null and handle it appropriately. "
        "Exceptions should only be used for truly exceptional cases, and invalid user input is "
        "expected. Returning null is more performant and gives the caller more flexibility.",
        "bad"
    ),
    # BAD: review of the review approach, not the code
    _d(
        "I appreciate that you added unit tests, but I think you should also add integration "
        "tests, end-to-end tests, property-based tests, and mutation tests. Our testing pyramid "
        "recommends a 70/20/10 split between unit/integration/e2e. Right now we're at maybe "
        "80/10/10 for this module.",
        "bad"
    ),
    # BAD: pedantic about something that doesn't matter
    _d(
        "The HTTP status code 201 (Created) is technically incorrect here because the resource "
        "already exists — you're updating it, not creating it. Strictly per RFC 7231 section "
        "6.3.2, 201 should only be returned when a new resource is created. The correct code "
        "for an update is 200 (OK) or 204 (No Content). Please fix for REST compliance.",
        "bad"
    ),
    # BAD: theoretically correct but practically irrelevant suggestion
    _d(
        "Using `Math.random()` for the shuffle algorithm introduces a subtle bias because "
        "the PRNG period doesn't evenly divide the number of possible permutations for arrays "
        "larger than ~2000 elements. For a truly uniform shuffle, use a cryptographic RNG. "
        "The bias is approximately 1 in 2^53 for typical array sizes.",
        "bad"
    ),
    # BAD: detailed comparison that misses the actual issue
    _d(
        "I compared the previous implementation with this refactored version and functionally "
        "they're identical. The old code used a `for` loop while this uses `map` + `filter`. "
        "The algorithmic complexity is the same: O(n) in both cases. The memory footprint is "
        "slightly higher due to intermediate array creation, but negligible. No concerns.",
        "bad"
    ),
    # BAD: suggests a fix that introduces a new bug
    _d(
        "The timezone handling is off. `new Date()` returns local time, but your API clients "
        "expect UTC. Fix this by subtracting `getTimezoneOffset()` from the date: "
        "`date.setMinutes(date.getMinutes() - date.getTimezoneOffset())`. This converts "
        "local time to UTC without any library dependencies.",
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
        use_merge=True,
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
