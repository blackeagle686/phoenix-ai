def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

result = fibonacci(10)

with open('test_parallel_output.txt', 'w') as f:
    f.write(str(result) + '\n')

print(f'The 10th Fibonacci number is {result} and has been saved to test_parallel_output.txt')