def extract_number_from_sentence(sentence):
    # Mapping of number words to numerical equivalents
    number_words = {
        "ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4, "FIVE": 5, "SIX": 6, "SEVEN": 7, 
        "EIGHT": 8, "NINE": 9, "TEN": 10, "ELEVEN": 11, "TWELVE": 12, "THIRTEEN": 13, 
        "FOURTEEN": 14, "FIFTEEN": 15, "SIXTEEN": 16, "SEVENTEEN": 17, "EIGHTEEN": 18, 
        "NINETEEN": 19, "TWENTY": 20, "THIRTY": 30, "FORTY": 40, "FIFTY": 50, 
        "SIXTY": 60, "SEVENTY": 70, "EIGHTY": 80, "NINETY": 90, "HUNDRED": 100, 
        "THOUSAND": 1000
    }

    def words_to_number(words):
        """Convert a sequence of number words into an integer."""
        total = 0
        current = 0
        for word in words:
            word_upper = word.upper()
            if word_upper in number_words:
                value = number_words[word_upper]
                if value == 100 or value == 1000:  # Handle multipliers
                    current *= value
                else:
                    current += value
            elif current > 0:
                total += current
                current = 0
        total += current
        return total

    # Split the sentence into words
    words = sentence.split()

    # Filter words that match the number words
    filtered_words = [word for word in words if word.upper() in number_words]

    # Convert the filtered words into a number
    return words_to_number(filtered_words)

def find_check_amount(text):
    chars = "$£"
    lines = text.splitlines()
    numbers = ["ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE",
 "TEN", "ELEVEN", "TWELVE", "THIRTEEN", "FOURTEEN", "FIFTEEN", "SIXTEEN", "SEVENTEEN", "EIGHTEEN", "NINETEEN",
 "TWENTY", "THIRTY", "FORTY", "FIFTY", "SIXTY", "SEVENTY", "EIGHTY", "NINETY",
 "HUNDRED", "THOUSAND"]

    # Loop through each line
    for i in range(len(lines)):
        line = lines[i]

        # Tokenize the line into words
        words = line.split()

        # Check each word or token in the line
        for word in words:
            if word.upper() in numbers:  # Check if the word matches
                return extract_number_from_sentence(line)
            elif any(c in word for c in chars) and len(word) > 4:  # Check for special chars
                return word

    return "Amount Not Found"

# Example Usage
sentence = "I have ONE HUNDRED and TWENTY THREE apples."
result = find_check_amount(sentence)
print(result)  # Output: 123
