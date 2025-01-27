import random
import re
import numpy as np
from sample_cot import BasicIntArithmeticTaskConfig, generate_task, sample_concurrent
import asyncio

def test_arithmetic_generation(const_terms: bool = False):
    rng = random.Random()

    cfg = BasicIntArithmeticTaskConfig(
        min_digits=1,
        max_digits=7,
        min_terms=2,
        max_terms=8,
        operators=["+", "-"]
    )
    counts = np.zeros((cfg.max_digits - cfg.min_digits + 1, cfg.max_terms - cfg.min_terms + 1), dtype=np.int32)
    num_digits_labels = list(range(cfg.min_digits, cfg.max_digits + 1))
    num_terms_labels = list(range(cfg.min_terms, cfg.max_terms + 1))

    for _ in range(10000):
        task, answer, num_terms, num_digits = generate_task(rng, cfg, const_terms)

        # Extract the arithmetic expression
        expression = task.replace("What is ", "").replace("Solve ", "").replace("Calculate ", "").replace(" =", "").replace("?", "").replace(" ", "")
        counts[num_digits-cfg.min_digits, num_terms-cfg.min_terms] += 1
        # Calculate actual result
        try:
            actual = eval(expression)
            assert str(actual) == answer, f"Mismatch: expected {answer}, got {actual}"
            if const_terms:
                terms = re.findall(r'\d+', expression)
                assert all(len(str(t)) == num_digits for t in terms), f"Constant term {num_digits} digits out of range: {task}"
            # print(f"✓ Test passed: Calculated {actual} = ", end="")
        except Exception as e:
            print(f"× Test failed: {e}: ", end="")
            print(f"{expression} ({num_terms} terms, {num_digits} digits)")
        # print(f"{expression} ({num_terms} terms, {num_digits} digits)")

        # Verify constraints
        assert num_terms >= cfg.min_terms and num_terms <= cfg.max_terms, f"Terms constraint violated: {num_terms}"
        # print(f"Number of terms: {num_terms} (within [{cfg.min_terms}, {cfg.max_terms}])")
    print("Counts (x: num_terms, y: num_digits):")
    print(f"{'':<15}", end="")
    for term in num_terms_labels:
        print(f"{term:<8}", end="")
    print('\n')
    for i, row in enumerate(counts):
        print(f"{num_digits_labels[i]:<15}", end="")
        for count in row:
            print(f"{count:<8}", end="")
        print()

async def test_sample_concurrent(uniform: bool = False):
    rng = random.Random(42)
    cfg = BasicIntArithmeticTaskConfig(
        min_digits=1,
        max_digits=7,
        min_terms=2,
        max_terms=8,
        operators=["+", "-"]
    )
    counts = np.zeros((cfg.max_digits - cfg.min_digits + 1, cfg.max_terms - cfg.min_terms + 1), dtype=np.int32)
    num_digits_labels = list(range(cfg.min_digits, cfg.max_digits + 1))
    num_terms_labels = list(range(cfg.min_terms, cfg.max_terms + 1))

    test_prompt = "Test prompt"
    n_samples = 10000

    # Call sample_concurrent with client=None
    results = await sample_concurrent(
        rng=rng,
        client=None,
        n=n_samples,
        task_cfg=cfg,
        developer_prompt=test_prompt,
        output_jsonl="dummy.jsonl",  # Not used when client is None
        sampling_params={"model": "test-model"},
        max_concurrent=1,
        api_type="test",
        uniform=uniform
    )

    # Verify results
    assert len(results) == n_samples, f"Expected {n_samples} results, got {len(results)}"

    for result in results:
        # Verify result structure
        assert "num_terms" in result
        assert "num_digits" in result
        assert "ground_truth" in result
        assert "input" in result

        # Verify constraints
        assert result["num_terms"] >= cfg.min_terms and result["num_terms"] <= cfg.max_terms

        counts[result["num_digits"]-cfg.min_digits, result["num_terms"]-cfg.min_terms] += 1
        #print(f"\n({result['num_terms']} terms, {result['num_digits']} digits) {result['input']}={result['ground_truth']}")
    print("\nCounts (x: num_terms, y: num_digits):")
    print(f"{'':<15}", end="")
    for term in num_terms_labels:
        print(f"{term:<8}", end="")
    print('\n')
    for i, row in enumerate(counts):
        print(f"{num_digits_labels[i]:<15}", end="")
        for count in row:
            print(f"{count:<8}", end="")
        print()

if __name__ == "__main__":
    print("Arithmetic generation:")
    test_arithmetic_generation()
    print("-"*10)
    print("Arithmetic generation (constant terms):")
    test_arithmetic_generation(const_terms=True)
    print("-"*10)
    print("Arithmetic sampling:")
    asyncio.run(test_sample_concurrent())
    print("-"*10)
    print("Arithmetic sampling (uniform):")
    asyncio.run(test_sample_concurrent(uniform=True))
