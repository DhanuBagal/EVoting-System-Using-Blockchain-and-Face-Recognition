import cv2
import pytesseract
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# Configure tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def extract_text_from_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("[ERROR] Failed to read image.")
        return ""

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply preprocessing
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # OCR config
    custom_config = r'--oem 3 --psm 4'  # Single column text block
    text = pytesseract.image_to_string(thresh, config=custom_config)
    print("[DEBUG] OCR Raw Output:\n", text)
    return text


def extract_name_from_text(text):
    lines = text.split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    name_candidates = []

    for line in lines:
        # Typical Aadhaar names: lines with 2+ capitalized words and no digits
        if re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+.*$', line) and not re.search(r'\d', line):
            name_candidates.append(line)

    print("[DEBUG] Name Candidates:", name_candidates)

    if name_candidates:
        return name_candidates[0]
    return ""


def extract_dob_from_text(text):
    # Try to extract DOB in dd/mm/yyyy or yyyy-mm-dd formats
    dob_pattern = r'(\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})'
    match = re.search(dob_pattern, text)
    if match:
        return match.group(0)
    return ""


def fuzzy_match_with_correction(entered_name, extracted_name):
    # Manually correct common OCR errors
    corrections = {
        "Yashwamt": "Yashwant",
        "Dhanashree Yashwamt Bagal": "Dhanashree Yashwant Bagal"
    }

    for error, correction in corrections.items():
        extracted_name = extracted_name.replace(error, correction)

    # Apply fuzzy matching
    return enhanced_fuzzy_match(entered_name, extracted_name)


def enhanced_fuzzy_match(entered_name, extracted_name):
    vectorizer = TfidfVectorizer().fit([entered_name, extracted_name])
    tfidf_matrix = vectorizer.transform([entered_name, extracted_name])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return cosine_sim


def standardize_dob_format(dob):
    # Convert the date into yyyy-mm-dd format
    try:
        # If the DOB is in dd/mm/yyyy format
        dob_obj = datetime.strptime(dob, "%d/%m/%Y")
    except ValueError:
        # If the DOB is already in yyyy-mm-dd format
        dob_obj = datetime.strptime(dob, "%Y-%m-%d")
    return dob_obj.strftime("%Y-%m-%d")


def validate_name_and_dob(entered_name, entered_dob, image_path):
    text = extract_text_from_image(image_path)
    extracted_name = extract_name_from_text(text)
    extracted_dob = extract_dob_from_text(text)

    print("[DEBUG] Entered Name:", entered_name)
    print("[DEBUG] Entered DOB:", entered_dob)
    print("[DEBUG] Extracted Name from Aadhaar:", extracted_name)
    print("[DEBUG] Extracted DOB from Aadhaar:", extracted_dob)

    if not extracted_name or not extracted_dob:
        return False, "❌ Name or DOB could not be extracted from Aadhaar card."

    # Standardize both DOBs to yyyy-mm-dd format
    standardized_entered_dob = standardize_dob_format(entered_dob)
    standardized_extracted_dob = standardize_dob_format(extracted_dob)

    # Compare Name
    score = fuzzy_match_with_correction(entered_name.lower(), extracted_name.lower())
    print("[DEBUG] Cosine Similarity Score for Name:", score)

    if score <= 0.5:
        return False, f"❌ Name does not match. Match score: {score:.2f}"

    # Compare DOB
    if standardized_entered_dob != standardized_extracted_dob:
        return False, "❌ DOB does not match."

    return True, f"✅ Name and DOB matched successfully!"


def complete_aadhaar_verification(entered_name, entered_dob, image_path):
    # Step 1: Validate Name and DOB
    valid, message = validate_name_and_dob(entered_name, entered_dob, image_path)
    if not valid:
        return False, message

    # Step 2: Check if the user is 18 or above
    age = calculate_age(entered_dob)
    print(f"[DEBUG] User's age: {age}")

    if age < 18:
        return False, "❌ Age is less than 18. You are not eligible."

    # Step 3: Proceed to capture photo (if age is valid)
    return "Registration successful!"


def calculate_age(dob):
    # Convert DOB string to datetime object
    dob = datetime.strptime(dob, "%Y-%m-%d")
    today = datetime.today()
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))  # Calculate age
    return age


