import re
import yaml
from typing import Optional, Dict, Any

class PromptProcessor:
    @staticmethod
    def create_summary_prompt(user_input: str) -> str:
        """Create prompt for the LLM to summarize user input."""
        return (
            "Based on the following input, please provide a YAML response in a code block with two keys:\n"
            "'statement': a concise statement (max 120 chars) rephrasing my text as a statement\n"
            "'confirmed': set to true ONLY if I confirm that the summary captures "
            "the intent. If any clarification is needed, set to false "
            f"Input: {user_input}"
        )

    @staticmethod
    def extract_yaml_content(response: str) -> Optional[Dict[str, Any]]:
        """Extract and parse YAML content from LLM response."""
        # Try to find content within code blocks first
        code_block_pattern = r'\`\`\`(?:yaml|yml)?\n([\s\S]*?)\`\`\`'
        code_blocks = re.findall(code_block_pattern, response, re.DOTALL)

        try:
            if code_blocks:
                parsed_yaml = yaml.safe_load(code_blocks[0])
            else:
                # If no code blocks found, try to parse the entire response as YAML
                parsed_yaml = yaml.safe_load(response)

            # If the parsed result is a list, merge all dictionaries in the list
            if isinstance(parsed_yaml, list):
                result = {}
                for item in parsed_yaml:
                    if isinstance(item, dict):
                        result.update(item)
                return result

            return parsed_yaml if isinstance(parsed_yaml, dict) else None

        except yaml.YAMLError:
            return None

    @staticmethod
    def clean_markdown(text: str) -> str:
        """Remove markdown syntax from text for TTS readability."""
        # Remove code blocks
        text = re.sub(r'```[\s\S]*?```', '', text)
        # Remove inline code
        text = re.sub(r'`[^`]*`', '', text)
        # Remove headers
        text = re.sub(r'#{1,6}\s.*\n', '', text)
        # Remove bold/italic
        text = re.sub(r'\*\*?(.*?)\*\*?', r'\1', text)
        # Remove links
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
        # Remove bullet points
        text = re.sub(r'^\s*[-*+]\s', '', text, flags=re.MULTILINE)
        return text.strip()
