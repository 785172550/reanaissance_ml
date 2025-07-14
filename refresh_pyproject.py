import toml
import sys
import os


def parse_requirements(req_file):
    deps = []
    with open(req_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                deps.append(line)
    return deps


def update_dependencies(pyproject_file, new_deps):
    if os.path.exists(pyproject_file):
        with open(pyproject_file, "r", encoding="utf-8") as f:
            data = toml.load(f)
    else:
        # 如果没有 pyproject.toml，初始化一个基本结构
        data = {
            "build-system": {
                "requires": ["setuptools>=64.0.0", "wheel"],
                "build-backend": "setuptools.build_meta",
            },
            "project": {
                "name": "Crawl4AI",
                "dynamic": ["version"],
                "description": "🚀🤖 Crawl4AI: Open-source LLM Friendly Web Crawler & scraper",
                "readme": "README.md",
                "requires-python": ">=3.9",
                "license": "Apache-2.0",
                "authors": [{"name": "Unclecode", "email": "unclecode@kidocode.com"}],
                "dependencies": [],
            },
        }

    # 确保有 project 部分
    if "project" not in data:
        data["project"] = {}
    # 更新 dependencies
    data["project"]["dependencies"] = new_deps

    with open(pyproject_file, "w", encoding="utf-8") as f:
        toml.dump(data, f)

    print(f"✅ 成功更新 {pyproject_file} 中的 dependencies！")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python update_deps.py requirements.txt")
        sys.exit(1)

    req_file = sys.argv[1]
    if not os.path.exists(req_file):
        print(f"❌ 文件不存在: {req_file}")
        sys.exit(1)

    deps = parse_requirements(req_file)
    update_dependencies("pyproject.toml", deps)
