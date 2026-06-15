import os

file_path = "scripts/run_joint_learning.py"
with open(file_path, "r") as f:
    content = f.read()

# 1. Update imports
content = content.replace(
    "from src.utils.translation import translate_rule",
    "from src.utils.translation import translate_rule, translate_to_mgr_syntax"
)

# 2. Add argument
content = content.replace(
    'parser.add_argument("--use_metrics", action="store_true", help="Include Support and Confidence as numerical features in the embedding.")',
    'parser.add_argument("--use_metrics", action="store_true", help="Include Support and Confidence as numerical features in the embedding.")\n    parser.add_argument("--use_mgr_syntax", action="store_true", help="Use MINE GRAPH RULE syntax for text translation instead of natural language.")'
)

# 3. Update translation execution
old_translation_block = """    print("Translating rules to natural language...")
    essential_cols = ['Anchor Label', 'Body', 'Head', 'Body Node Names', 'Head Node Names']
    df = df.dropna(subset=essential_cols)
    
    X_text = df.apply(translate_rule, axis=1).values"""

new_translation_block = """    essential_cols = ['Anchor Label', 'Body', 'Head', 'Body Node Names', 'Head Node Names']
    df = df.dropna(subset=essential_cols)
    
    if args.use_mgr_syntax:
        print("Translating rules to MINE GRAPH RULE syntax...")
        X_text = df.apply(lambda row: translate_to_mgr_syntax(row, f"Rule_{row.name}"), axis=1).values
    else:
        print("Translating rules to natural language...")
        X_text = df.apply(translate_rule, axis=1).values"""

content = content.replace(old_translation_block, new_translation_block)

with open(file_path, "w") as f:
    f.write(content)

print("MGR syntax flag added to run_joint_learning.py")
