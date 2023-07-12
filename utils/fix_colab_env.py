import pathlib
import json

ipynb_dir = pathlib.Path("d2l-tensorflow-colab")

def fix_colab_env(path):
    with open(path, "r") as f:
        try:
            data = json.load(f)
        except json.decoder.JSONDecodeError:
            print(f"Failed to parse {path}")
            return

    cells = data["cells"]

    if len(cells) >= 2:
        second_cell = "\n".join(cells[1]["source"])
        first_cell = "\n".join(cells[0]["source"])
        if "additional libraries" in first_cell and "pip install" in second_cell:
            cells.insert(1, {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "**Restart the runtime before run the cell below**"
                ]
            })
            cells.insert(1, {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "!pip install setuptools==65.5.0 \"wheel<0.40.0\""
                ]
            })

            with open(path, "w") as f:
                json.dump(data, f, indent=2)

            return

    print(f"No need to fix {path}")


if __name__ == "__main__":
    for path in ipynb_dir.glob("**/*.ipynb"):
        fix_colab_env(path)
