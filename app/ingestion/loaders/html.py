from bs4 import BeautifulSoup
import logfire


def parse_html(file_path: str) -> str:
    """
    Parse HTML content using BeautifulSoup
    Cleans Script , CSS style tags and extracts text content from body
    """
    with logfire.span("HTML Parsing", filename=file_path):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            soup = BeautifulSoup(content, "html.parser")
            
            # Remove Junk (Scripts, Styles, Metadata)
            for script in soup(["script", "style", "meta", "noscript"]):
                script.decompose()

            # Exteract Text
            text = soup.get_text(separator="\n")

            # Clean Whitespace (Collapse multiple newlines)
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text_clean = '\n'.join(chunk for chunk in chunks if chunk)

            logfire.info(f"Successfully parsed HTML file with {len(text_clean)} characters")
            return text_clean
        except Exception as e:
            logfire.error(f"Error parsing HTML file: {e}")
            raise e