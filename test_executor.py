from src.utils.executor import SafeCodeExecutor

executor = SafeCodeExecutor()
code = "import pandas as pd"
result = executor.execute(code)
print("SUCCESS:", result.success)
if not result.success:
    print(f"ERROR: {result.error_type} - {result.error_message}")
