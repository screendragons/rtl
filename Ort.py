import string
import re

class Ort:
    def to_string(path):
        with open(path, "r", encoding = "ISO-8859-1") as f:
            filtered_lines = []
            for line in f:
                if re.search(r'"[a-zA-Z0-9.\s!?]*"', line):
                    filtered_lines.append(line.strip())


            # Remove the first 6 lines and the last 4 lines
            filtered_lines = filtered_lines[6:-4]

            all_text = ""

            # Print the filtered text
            for line in filtered_lines:
                # Skip string if empty
                if line == '""':
                    continue

                # add a space after each line
                line += " "

                all_text += line

            # Remove all double quotes from the text
            all_text = all_text.replace('"', '')

             # Remove all punctuation from the text
            all_text = all_text.translate(str.maketrans("", "", string.punctuation))


            return all_text