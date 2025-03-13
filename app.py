from flask import Flask, jsonify, request
from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
import os
import base64, time
import numpy as np
import cv2,dlib
import Blockchain
import re


# Define paths for the models
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
FACE_RECOGNITION_MODEL_PATH = "dlib_face_recognition_resnet_model_v1.dat"

# Load the dlib models (only once)
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
recognizer = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL_PATH)

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:Afling%40123@localhost:3306/evoting_system'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Load Haar Cascade model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    userid = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(50), nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    area = db.Column(db.String(100), nullable=False)


class Party(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    party_name = db.Column(db.String(100), nullable=False)
    party_photo = db.Column(db.LargeBinary, nullable=False)


class Candidate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    candidate_name = db.Column(db.String(100), nullable=False)
    party_id = db.Column(db.Integer, db.ForeignKey('party.id'), nullable=False)
    area = db.Column(db.String(100), nullable=False)
    party = db.relationship('Party', backref=db.backref('candidates', lazy=True))


# Models for Vote
class Vote(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    candidate = db.Column(db.String(100), nullable=False)
    blockchain_hash = db.Column(db.String(255), nullable=False)  # Ensure this is NOT NULL



# Initialize the blockchain
blockchain = Blockchain.Blockchain()

# Check if user exists in the blockchain (your provided function)
def checkUser(name):
    flag = 0
    for i in range(len(blockchain.chain)):
        if i > 0:  # Skip the genesis block
            b = blockchain.chain[i]
            transactions = b['transactions']  # Get the transactions list

            # Check if the transactions list is not empty before accessing it
            if len(transactions) > 0:
                data = transactions[0]  # Access the first transaction
                voter_id = data['voter_id']  # Access 'voter_id' from the dictionary
                if voter_id == name:  # Compare voter_id with the input 'name'
                    flag = 1
                    break
            else:
                print(f"No transactions found in block {i}.")
    return flag




@app.route('/')
def home():
    # Render home page
    return render_template('index.html')



@app.route('/register', methods=['GET', 'POST'])
def register():
    # User registration
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        userid = request.form['userid']
        password = request.form['password']
        phone = request.form['phone']
        area = request.form['area']

        # Print the received data for debugging
        print("Received registration data:")
        print(f"Name: {name}")
        print(f"Email: {email}")
        print(f"User ID: {userid}")
        print(f"Password: {password}")
        print(f"Phone: {phone}")
        print(f"Area: {area}")

        # Validate password
        password_pattern = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'
        if not re.match(password_pattern, password):
            return "Password must be at least 8 characters long and contain at least one uppercase letter, one lowercase letter, one number, and one special character."

        # Validate phone number
        if len(phone) != 10 or not phone.isdigit():
            return "Phone number must be exactly 10 digits and contain only numbers."

        # Create and save new user in the database
        try:
            new_user = User(name=name, email=email, userid=userid, password=password, phone=phone, area=area)
            db.session.add(new_user)
            db.session.commit()
            print("User registered successfully.")

            # Set session and redirect to capture photo
            session['userid'] = userid
            print("User ID saved in session. Redirecting to capture photo.")
            return redirect(url_for('capture_photo'))

        except Exception as e:
            print(f"Error during registration: {e}")
            return "Registration failed. Please try again."

    print("Rendering registration form.")
    return render_template('register.html')



@app.route('/capture_photo', methods=['GET', 'POST'])
def capture_photo():
    # Print statement to indicate that the capture photo page is being accessed
    print("Accessing capture photo page.")
    return render_template('capture_photo.html')  # Assume you have a form to capture the photo


@app.route('/save_photo', methods=['POST'])
def save_photo():
    if 'userid' in session:
        photo_data = request.form['photo']
        print("Received photo data for user:", session['userid'])

        if not photo_data or ',' not in photo_data:
            print("Invalid photo data received.")
            return "Invalid photo data", 400

        try:
            # Split and decode the base64 data
            photo_base64 = photo_data.split(',')[1]
            voter_id = session['userid']  # Get voter ID from session
            voter_dir = f"static/photo/user/{voter_id}"
            os.makedirs(voter_dir, exist_ok=True)  # Ensure the directory exists
            print(f"Directory created for voter ID: {voter_id} at {voter_dir}")

            max_photos = 5
            for count in range(max_photos):
                # Decode the base64 image data
                image_data = base64.b64decode(photo_base64)
                image_array = np.frombuffer(image_data, dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

                # Detect faces in the image
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

                print(f"Detected {len(faces)} faces in the photo.")

                if len(faces) == 0:
                    print("No face detected.")
                    return "No face detected", 400

                # Crop and save the detected face
                for (x, y, w, h) in faces:
                    face_image = image[y:y+h, x:x+w]
                    img_path = os.path.join(voter_dir, f"face_{count}.jpg")
                    cv2.imwrite(img_path, face_image)
                    print(f"Saved face image to {img_path}")
                    break  # Only crop the first detected face for this photo

            print("Photos saved successfully. Redirecting to login page.")
            # Redirect to the login page after saving the photos
            return redirect(url_for('login'))

        except (IndexError, ValueError) as e:
            print(f"Error processing photo: {e}")
            return f"Error processing photo: {e}", 400

    print("Session expired or user ID not found.")
    return "Session expired or user ID not found", 400


@app.route('/login', methods=['GET', 'POST'])
def login():
    # User login
    if request.method == 'POST':
        userid = request.form['username']
        password = request.form['password']
        role = request.form['role']

        print(f"Attempting to log in user: {userid} with role: {role}")

        # Admin login
        if userid == "admin" and password == "admin" and role == "admin":
            session['userid'] = userid
            session['role'] = role
            print(f"Admin {userid} logged in successfully.")
            return redirect(url_for('admin_dashboard'))

        # Voter login
        user = User.query.filter_by(userid=userid, password=password).first()
        if user:
            session['userid'] = userid
            session['role'] = role
            print(f"Voter {userid} logged in successfully.")
            if role != 'admin':
                return redirect(url_for('validate_voter'))
        else:
            print("Invalid credentials provided.")
            return "Invalid credentials. Please try again."

    return render_template('login.html')


@app.route('/logout')
def logout():
    # User logout
    userid = session.get('userid', None)
    session.pop('userid', None)
    session.pop('role', None)
    print(f"User {userid} logged out.")
    return redirect(url_for('home'))


@app.route('/validate_voter', methods=['GET', 'POST'])
def validate_voter():
    # Photo validation for voter
    if request.method == 'POST' and 'userid' in session:
        userid = session['userid']
        photo_data = request.form.get('photo')

        print(f"Validating photo for user: {userid}")

        # Validate photo data
        if not photo_data or ',' not in photo_data:
            print("Invalid photo data received.")
            return render_template('validate_voter.html', message="Invalid photo data.")

        try:
            # Validate the voter face using saved data
            result, message = validate_voter_face(photo_data, userid)
            if result:
                if checkUser(session['userid']):
                    return redirect(url_for('voter_dashboard'))  # Ensure this points to the dashboard route

                    # Proceed to voting page if not already voted
                return redirect(url_for('voter_dashboard'))

            else:
                print(f"Photo validation failed for user: {userid}. Message: {message}")
                return render_template('validate_voter.html', message=message)
        except Exception as e:
            print(f"An error occurred during photo validation for user: {userid}. Error: {e}")
            return render_template('validate_voter.html', message="An error occurred during validation.")

    # Render the validation page on GET request or if no userid in session
    return render_template('validate_voter.html')


def validate_voter_face(photo_data, userid):
    print("Face recognition process started.")
    try:
        # Decode the base64 photo data
        photo_base64 = photo_data.split(',')[1]
        image_data = base64.b64decode(photo_base64)
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        captured_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Path to user's registered face images
        voter_dir = f'static/photo/user/{userid}'
        if not os.path.exists(voter_dir):
            print(f"No registered face found for user: {userid}.")
            return False, "No registered face found for this user."

        known_faces = []
        for img_name in os.listdir(voter_dir):
            print(f"Loading image: {img_name}")
            img_path = os.path.join(voter_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue

            # Convert image to grayscale for face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)  # Using your face detector
            if len(faces) > 0:
                print("Face detected in:", img_name)
                shape = shape_predictor(gray, faces[0])  # Shape predictor
                face_descriptor = np.array(recognizer.compute_face_descriptor(img, shape))  # Compute face descriptor
                known_faces.append(face_descriptor)
            else:
                print("No face detected in:", img_name)

        if not known_faces:
            print(f"No faces found in the registered images for user: {userid}.")
            return False, "No faces found in the registered images."

        # Process the captured image
        gray_captured = cv2.cvtColor(captured_image, cv2.COLOR_BGR2GRAY)
        faces_captured = detector(gray_captured)
        if len(faces_captured) == 0:
            print(f"No face detected in the captured image for user: {userid}.")
            return False, "No face detected in the captured image."

        # Compute descriptor for the captured face
        shape_captured = shape_predictor(gray_captured, faces_captured[0])
        captured_face_descriptor = np.array(recognizer.compute_face_descriptor(captured_image, shape_captured))

        # Calculate distances between the known faces and the captured face
        distances = np.linalg.norm(np.array(known_faces) - captured_face_descriptor, axis=1)
        min_distance = np.min(distances)
        print(f"Minimum distance for user {userid}: {min_distance}")

        # Check if the minimum distance is below the 0.6 threshold
        if min_distance < 0.6:
            print(f"Face validation successful for user {userid}.")
            return True, "Face validated successfully."
        else:
            print(f"Face validation failed for user {userid}. No match found within threshold.")
            return False, "Face does not match."

    except Exception as e:
        print(f"Error during face validation for user: {userid}. Error: {e}")
        return False, f"Error during face validation: {e}"


@app.route('/voter_dashboard')
def voter_dashboard():
    if 'userid' in session and session['role'] == 'voter':
        user = User.query.filter_by(userid=session['userid']).first()
        candidates = Candidate.query.join(Party).filter(Candidate.area == user.area).add_columns(
            Party.party_name, Party.party_photo, Candidate.candidate_name).all()

        # Pass the user ID (userid) to the template
        return render_template('voter_dashboard.html', candidates=candidates, voter_id=user.userid)

    return redirect(url_for('login'))


@app.route('/admin_dashboard')
def admin_dashboard():
    if 'userid' in session and session['role'] == 'admin':
        return render_template('admin_dashboard.html')
    return redirect(url_for('login'))


@app.route('/add_party', methods=['GET', 'POST'])
def add_party():
    if 'userid' in session and session['role'] == 'admin':
        if request.method == 'POST':
            party_id = request.form.get('party_id')
            party_name = request.form['party_name']
            party_photo = request.files['party_photo'].read()
            candidate_name = request.form['candidate_name']
            area = request.form['area']

            if party_id:  # Update existing party
                existing_party = Party.query.get(party_id)
                if existing_party:
                    existing_party.party_name = party_name
                    existing_party.party_photo = party_photo

                    existing_candidate = Candidate.query.filter_by(party_id=party_id, area=area).first()
                    if existing_candidate:
                        existing_candidate.candidate_name = candidate_name
                    else:
                        new_candidate = Candidate(candidate_name=candidate_name, party_id=party_id, area=area)
                        db.session.add(new_candidate)

                    db.session.commit()
                    message = "Party and candidate details updated successfully."
                else:
                    message = "Party not found."
            else:  # Add new party
                new_party = Party(party_name=party_name, party_photo=party_photo)
                db.session.add(new_party)
                db.session.commit()

                new_candidate = Candidate(candidate_name=candidate_name, party_id=new_party.id, area=area)
                db.session.add(new_candidate)
                db.session.commit()

                message = "Party and candidate added successfully."

            all_candidates = Candidate.query.join(Party).add_columns(Candidate.candidate_name, Candidate.area,
                                                                     Party.party_name, Candidate.party_id).all()

            return render_template('add_party.html', all_candidates=all_candidates, message=message)

        all_candidates = Candidate.query.join(Party).add_columns(Candidate.candidate_name, Candidate.area,
                                                                 Party.party_name, Candidate.party_id).all()
        return render_template('add_party.html', all_candidates=all_candidates)

    return redirect(url_for('login'))

@app.route('/cast_vote', methods=['POST'])
def cast_vote():
    # Retrieve form data
    voter_id = request.form.get('voter_id')
    party_name = request.form.get('party_name')

    required = [voter_id, party_name]
    if not all(required):
        return 'Missing values', 400

    # Create a new transaction (vote)
    transaction_index = blockchain.new_transaction(voter_id, party_name)

    # Mine a new block to store the vote
    last_block = blockchain.last_block
    proof = blockchain.proof_of_work(last_block['proof'])
    new_block = blockchain.new_block(proof, blockchain.hash(last_block))

    # Retrieve details of the newly created block
    block_hash = blockchain.hash(new_block)
    timestamp = new_block['timestamp']

    # Return the details of the vote transaction and block hash
    response = {
        'message': 'Vote successfully cast and recorded in the blockchain!',
        'block_hash': block_hash,
        'voter_id': voter_id,
        'party_name': party_name,
        'timestamp': timestamp
    }

    return jsonify(response), 201

# Mining route (mine a new block)
@app.route('/mine', methods=['GET'])
def mine():
    last_block = blockchain.last_block
    last_proof = last_block['proof']
    proof = blockchain.proof_of_work(last_proof)

    # Forge the new Block by adding it to the chain
    block = blockchain.new_block(proof)

    response = (
        "New Block Forged\n"  
        f"Index: {block['index']}\n"  
        f"Transactions: {block['transactions']}\n"  
        f"Proof: {block['proof']}\n"  
        f"Previous Hash: {block['previous_hash']}\n"
    )
    return response, 200

# Route to view the entire blockchain
@app.route('/chain', methods=['GET'])
def full_chain():
    chain_length = len(blockchain.chain)
    chain_info = "\n".join([str(block) for block in blockchain.chain])

    response = (
        f"Chain Length: {chain_length}\n"  
        "Chain:\n" + chain_info
    )
    return response, 200

# Admin route to view votes (fetches the blockchain's transactions)
@app.route('/view_votes', methods=['GET'])
def view_votes():
    # Retrieve all the votes stored in the blockchain
    votes = blockchain.get_votes()

    # Check if votes are empty
    if not votes:
        return jsonify({"message": "No votes found."}), 200

    # Format the votes for display
    formatted_votes = [
        f"Voter ID: {vote['voter_id']}, Party: {vote['party_name']}, "
        f"Timestamp: {vote['timestamp']}, Block Hash: {vote['block_hash']}"
        for vote in votes
    ]

    response = {
        "message": "Here are all the votes",
        "votes": formatted_votes
    }

    return jsonify(response), 200




# Ensure tables are created when the application starts
with app.app_context():
    db.create_all()


if __name__ == '__main__':
    app.run(debug=True)