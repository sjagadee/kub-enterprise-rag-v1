import logfire


def parse_text(file_path: str):
    """
    Parse plain text files
    """

    with logfire.span("Text parsing", filename=file_path):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            logfire.info(f"Successfully parsed text file with {len(text)} characters")
            return text
        except Exception as e:
            logfire.error(f"Error parsing text file: {e}")
            raise e
