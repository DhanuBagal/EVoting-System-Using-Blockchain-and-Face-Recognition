from flask import Flask, jsonify, request, render_template,flash, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
import os
import re
import cv2
import dlib
import numpy as np
import Blockchain
import base64
from werkzeug.utils import secure_filename
from sqlalchemy import text,UniqueConstraint,func, and_, or_
from aadhaar_verification import complete_aadhaar_verification
from scipy.spatial import distance
import json
from datetime import datetime


# Define paths for the models
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
FACE_RECOGNITION_MODEL_PATH = "dlib_face_recognition_resnet_model_v1.dat"

# Load the dlib models (only once)
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
recognizer = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL_PATH)

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

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
    email = db.Column(db.String(120), unique=True, nullable=False)
    userid = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    phone = db.Column(db.String(10), nullable=False)
    area = db.Column(db.String(100), nullable=False)
    dob = db.Column(db.String(10), nullable=False)  # Format: YYYY-MM-DD
    aadhaar_photo = db.Column(db.String(255), nullable=False)  # File path to Aadhaar photo
    profile_photo = db.Column(db.String(255), nullable=True)  # File path to profile photo
    created_at = db.Column(db.DateTime, default=datetime.utcnow)  # Automatically set the creation timestamp

class Temp_Users(db.Model):
    userid = db.Column(db.String(50), primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    password = db.Column(db.String(255), nullable=False)
    phone = db.Column(db.String(10), nullable=False)
    area = db.Column(db.String(100), nullable=False)
    dob = db.Column(db.String(10), nullable=False)
    aadhaar_photo = db.Column(db.String(255), nullable=False)  # Aadhaar photo file path


class Party(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    party_name = db.Column(db.String(100), nullable=False)
    logo_filename = db.Column(db.String(255))  # Filename of logo in /static/party_logos/


class Candidate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    candidate_name = db.Column(db.String(100), nullable=False)
    party_id = db.Column(db.Integer, db.ForeignKey('party.id'), nullable=False)
    area = db.Column(db.String(100), nullable=False)
    party = db.relationship('Party', backref=db.backref('candidates', lazy=True))


class Election(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    start_time = db.Column(db.Time, nullable=False)
    end_time = db.Column(db.Time, nullable=False)

    # Add a composite unique constraint for start_time and end_time
    __table_args__ = (
        UniqueConstraint('start_time', 'end_time', name='_start_end_time_uc'),
    )



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
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        userid = request.form['userid']
        password = request.form['password']
        phone = request.form['phone']
        area = request.form['area']
        dob = request.form['dob']
        aadhaar_photo = request.files['aadhaar_photo']

        userid_lower = userid.lower().strip()
        name_lower = name.lower().strip()
        email_lower = email.lower().strip()

        existing_user = User.query.filter(
            (func.lower(User.userid) == userid_lower) |
            (func.lower(User.name) == name_lower) |
            (func.lower(User.email) == email_lower)
        ).first()

        temp_user_query = text("""
            SELECT * FROM temp_users 
            WHERE LOWER(userid) = :userid OR LOWER(name) = :name OR LOWER(email) = :email
        """)
        temp_user_result = db.session.execute(temp_user_query, {
            'userid': userid_lower,
            'name': name_lower,
            'email': email_lower
        }).first()

        if existing_user or temp_user_result:
            flash("User with same ID, name, or email already exists. Please try another.", "error")
            return redirect(url_for('register'))

        password_pattern = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'
        if not re.match(password_pattern, password):
            flash("Weak password format. Password must be at least 8 characters long and include uppercase, lowercase, number, and special character.", "error")
            return redirect(url_for('register'))

        if len(phone) != 10 or not phone.isdigit():
            flash("Invalid phone number. Must be exactly 10 digits.", "error")
            return redirect(url_for('register'))

        if aadhaar_photo:
            ext = os.path.splitext(aadhaar_photo.filename)[1].lower()
            if ext not in ['.jpg', '.jpeg', '.png']:
                flash("Invalid Aadhaar photo format. Only JPG, JPEG, and PNG are allowed.", "error")
                return redirect(url_for('register'))

            filename = f"{secure_filename(userid)}_aadhaar{ext}"
            aadhaar_photo_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            aadhaar_photo.save(aadhaar_photo_path)

            verification_result = complete_aadhaar_verification(name, dob, aadhaar_photo_path)

            if verification_result != "Registration successful!":
                flash(verification_result, "error")
                return redirect(url_for('register'))

            conn = db.engine.connect()
            conn.execute(
                text('''INSERT INTO temp_users (name, email, userid, password, phone, area, dob, aadhaar_photo)
                        VALUES (:name, :email, :userid, :password, :phone, :area, :dob, :aadhaar_photo)'''),
                {
                    'name': name,
                    'email': email,
                    'userid': userid,
                    'password': password,
                    'phone': phone,
                    'area': area,
                    'dob': dob,
                    'aadhaar_photo': filename
                }
            )
            conn.commit()
            conn.close()

            session['registration_data'] = {
                'name': name,
                'email': email,
                'userid': userid,
                'password': password,
                'phone': phone,
                'area': area,
                'dob': dob,
                'aadhaar_photo': filename
            }

            flash("Registration successful. Proceed to capture photo.", "success")
            return redirect(url_for('capture_photo'))

    return render_template('register.html')


@app.route('/capture_photo', methods=['GET', 'POST'])
def capture_photo():
    # Print statement to indicate that the capture photo page is being accessed
    print("Accessing capture photo page.")
    return render_template('capture_photo.html')  # Assume you have a form to capture the photo

from flask import redirect, url_for  # make sure these are imported at the top

@app.route('/save_photo', methods=['POST'])
def save_photo():
    if 'registration_data' not in session:
        return "No session data found", 400

    voter_id = str(session['registration_data']['userid']).strip()
    if not voter_id:
        return "Voter ID missing in session", 400

    file = request.files.get('photo')
    if not file:
        cleanup_temp_user(voter_id)
        return redirect(url_for('register'))

    try:
        voter_dir = os.path.join("static", "photo", "user", voter_id)
        os.makedirs(voter_dir, exist_ok=True)
        file_path = os.path.join(voter_dir, f"photo_{voter_id}.png")
        file.save(file_path)

        image = cv2.imread(file_path)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            cleanup_temp_user(voter_id)
            return redirect(url_for('register'))

        for (x, y, w, h) in faces:
            face_image = image[y:y + h, x:x + w]
            face_image_path = os.path.join(voter_dir, f"{voter_id}_face.png")
            cv2.imwrite(face_image_path, face_image)
            break

        with db.engine.connect() as conn:
            result = conn.execute(
                text("SELECT * FROM temp_users WHERE LOWER(TRIM(userid)) = :userid"),
                {"userid": voter_id.lower()}
            )
            temp_user_row = result.fetchone()

            if not temp_user_row:
                cleanup_temp_user(voter_id)
                return redirect(url_for('register'))

            user_data = dict(temp_user_row._mapping)

        new_user = User(
            name=user_data['name'],
            email=user_data['email'],
            userid=user_data['userid'],
            password=user_data['password'],
            phone=user_data['phone'],
            area=user_data['area'],
            dob=user_data['dob'],
            aadhaar_photo=user_data['aadhaar_photo'],
            profile_photo=f"{voter_id}_face.png"
        )

        db.session.add(new_user)
        db.session.flush()
        db.session.execute(
            text("DELETE FROM temp_users WHERE LOWER(TRIM(userid)) = :userid"),
            {"userid": voter_id.lower()}
        )
        db.session.commit()

        return redirect(url_for('login'))

    except Exception as e:
        db.session.rollback()
        cleanup_temp_user(voter_id)
        return redirect(url_for('register'))

def cleanup_temp_user(voter_id):
    try:
        db.session.execute(
            text("DELETE FROM temp_users WHERE LOWER(TRIM(userid)) = :userid"),
            {"userid": voter_id.lower()}
        )
        db.session.commit()
        print(f"Cleaned up temp user {voter_id}")
    except:
        db.session.rollback()
        print(f"Failed to clean up temp user {voter_id}")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        userid = request.form['username']
        password = request.form['password']
        role = request.form['role']

        print(f"Attempting to log in user: {userid} with role: {role}")

        if userid == "admin" and password == "Admin@1234" and role == "admin":
            session['userid'] = userid
            session['role'] = role
            print(f"Admin {userid} logged in successfully.")
            flash("Admin logged in successfully!", "success")
            return redirect(url_for('admin_dashboard'))

        user = User.query.filter_by(userid=userid, password=password).first()
        if user:
            session['userid'] = userid
            session['role'] = role
            print(f"Voter {userid} logged in successfully.")
            if role != 'admin':
                flash("Voter logged in successfully!", "success")
                return redirect(url_for('validate_voter'))
        else:
            print("Invalid credentials provided.")
            flash("Invalid credentials. Please try again.", "danger")
            return redirect(url_for('login'))

    return render_template('login.html')


@app.route('/logout')
def logout():
    # User logout
    userid = session.get('userid', None)
    session.pop('userid', None)
    session.pop('role', None)
    print(f"User {userid} logged out.")
    return redirect(url_for('home'))



# Function to compute the Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function for liveness detection (blink detection)
def check_liveness(captured_image, blink_count=2, time_window=10):
    start_time = datetime.now()  # Using datetime for the start time
    blinks_detected = 0

    while (datetime.now() - start_time).seconds < time_window:  # Compare current time with start_time
        # Convert to grayscale
        gray_captured = cv2.cvtColor(captured_image, cv2.COLOR_BGR2GRAY)
        faces_captured = detector(gray_captured)

        if len(faces_captured) == 0:
            return False, "No face detected in the captured image."

        # Get facial landmarks
        shape_captured = shape_predictor(gray_captured, faces_captured[0])
        left_eye = [(shape_captured.part(i).x, shape_captured.part(i).y) for i in range(36, 42)]
        right_eye = [(shape_captured.part(i).x, shape_captured.part(i).y) for i in range(42, 48)]

        # Calculate EAR (Eye Aspect Ratio) for blink detection
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # If EAR is below the threshold, it indicates a blink
        EAR_THRESHOLD = 0.2
        if ear < EAR_THRESHOLD:
            blinks_detected += 1

        # If we have detected the required number of blinks, stop the check
        if blinks_detected >= blink_count:
            return True, f"Liveness detected with {blinks_detected} blinks."

    return False, f"Liveness not detected. Only {blinks_detected} blinks detected."

# Function to validate voter face
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
            return False, "No registered face found for this user."

        known_faces = []
        for img_name in os.listdir(voter_dir):
            img_path = os.path.join(voter_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Convert image to grayscale for face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            if len(faces) > 0:
                shape = shape_predictor(gray, faces[0])
                face_descriptor = np.array(recognizer.compute_face_descriptor(img, shape))
                known_faces.append(face_descriptor)

        if not known_faces:
            return False, "No faces found in the registered images."

        # Check for liveness (blink detection)
        liveness_result, liveness_message = check_liveness(captured_image)
        if not liveness_result:
            return False, liveness_message

        # Process the captured image for face recognition
        gray_captured = cv2.cvtColor(captured_image, cv2.COLOR_BGR2GRAY)
        faces_captured = detector(gray_captured)
        if len(faces_captured) == 0:
            return False, "No face detected in the captured image."

        shape_captured = shape_predictor(gray_captured, faces_captured[0])
        captured_face_descriptor = np.array(recognizer.compute_face_descriptor(captured_image, shape_captured))

        # Calculate distances between the known faces and the captured face
        distances = np.linalg.norm(np.array(known_faces) - captured_face_descriptor, axis=1)
        min_distance = np.min(distances)

        # Check if the minimum distance is below the threshold
        if min_distance < 0.6:
            return True, "Face validated successfully."
        else:
            return False, "Face does not match."

    except Exception as e:
        return False, f"Error during face validation: {e}"

# Route for validating the voter
@app.route('/validate_voter', methods=['GET', 'POST'])
def validate_voter():
    if request.method == 'POST':  # This block is for when the form is submitted
        if 'userid' in session:
            userid = session['userid']
            photo_data = request.form.get('photo')

            if not photo_data or ',' not in photo_data:
                return render_template('validate_voter.html', message="Invalid photo data.")

            result, message = validate_voter_face(photo_data, userid)
            if result:
                flash("Validation successful! Redirecting to your dashboard.", "success")
                return redirect(url_for('voter_dashboard'))
            else:
                return render_template('validate_voter.html', message=message)

        return render_template('validate_voter.html', message="Session expired. Please login again.")

    # This block handles the GET request (initial page load)
    return render_template('validate_voter.html')



from datetime import datetime, date
from flask import flash, render_template, redirect, url_for

@app.route('/voter_dashboard')
def voter_dashboard():
    if 'userid' in session and session['role'] == 'voter':
        user = User.query.filter_by(userid=session['userid']).first()
        voter_area = user.area

        # Get today's date and current time
        today = date.today()
        now = datetime.now().time()

        # Query for election scheduled for today and within the time window
        election = Election.query.filter_by(date=today).filter(
            Election.start_time <= now, Election.end_time >= now
        ).first()

        if election:
            try:
                # Election is live, show candidates
                candidates = Candidate.query.join(Party).filter(Candidate.area == voter_area).add_columns(
                    Party.party_name, Party.logo_filename, Candidate.candidate_name, Candidate.area).all()
                election_status = "Election is live, you can vote now!"
                flash("Election is live, you can vote now!", "success")
                return render_template('voter_dashboard.html', candidates=candidates, voter_id=user.userid, election_status=election_status)
            except Exception as e:
                election_status = "There was an issue with fetching candidates."
                flash("Error while fetching candidates: " + str(e), "danger")
        else:
            election_status = "Election is either not scheduled today or not open right now."
            flash("Election is either not scheduled today or not open right now.", "warning")

        return render_template('voter_dashboard.html', election_status=election_status)

    return redirect(url_for('login'))

@app.route('/add_party', methods=['GET', 'POST'])
def add_party():
    if 'userid' in session and session['role'] == 'admin':
        # List of Solapur talukas
        solapur_talukas = ["Solapur North", "Solapur South", "Akkalkot", "Barshi", "Karmala", "Madha", "Mangalwedha", "Malshiras", "Pandharpur", "Sangola"]

        # Get all parties from DB
        maharashtra_parties = Party.query.all()
        parties_logo_map = {str(p.id): p.logo_filename for p in maharashtra_parties}

        if request.method == 'POST':
            party_id = request.form.get('party_id')  # The party ID from the form
            selected_party_id = request.form['party_name']  # The party ID selected by the user
            candidate_name = request.form['candidate_name']
            area = request.form['area']

            # Handle logo file (if provided)
            if 'party_photo' in request.files:
                logo_file = request.files['party_photo']
                if logo_file:
                    # Secure the filename and save it
                    filename = secure_filename(logo_file.filename)
                    logo_file.save(os.path.join('static/party_logos', filename))
                else:
                    filename = None
            else:
                filename = None

            # Check if candidate name already exists in the database
            if Candidate.query.filter_by(candidate_name=candidate_name).first():
                message = "Candidate name already exists."
                message_type = "error"
            # Check if the same party already has a candidate in the selected area
            elif Candidate.query.filter_by(party_id=selected_party_id, area=area).first():
                message = "This party already has a candidate in the selected area."
                message_type = "error"
            else:
                # If party_id is present, update the existing candidate and party details
                if party_id:  # Update existing candidate
                    existing_party = Party.query.get(party_id)
                    if existing_party:
                        existing_party.party_name = request.form['party_name']
                        if filename:
                            existing_party.logo_filename = filename

                        # Check if the candidate already exists for the party and area
                        existing_candidate = Candidate.query.filter_by(party_id=party_id, area=area).first()
                        if existing_candidate:
                            existing_candidate.candidate_name = candidate_name
                        else:
                            new_candidate = Candidate(candidate_name=candidate_name, party_id=party_id, area=area)
                            db.session.add(new_candidate)
                        db.session.commit()
                        message = "Party and candidate details updated successfully."
                        message_type = "success"
                    else:
                        message = "Party not found."
                        message_type = "error"
                else:  # Add new candidate under an existing party
                    # If party doesn't exist, retrieve it from the DB based on selected_party_id
                    party = Party.query.get(selected_party_id)
                    if party:
                        # Only add the candidate to the existing party
                        new_candidate = Candidate(candidate_name=candidate_name, party_id=party.id, area=area)
                        db.session.add(new_candidate)
                        db.session.commit()
                        message = "Candidate added successfully."
                        message_type = "success"
                    else:
                        message = "Selected party does not exist."
                        message_type = "error"

        all_candidates = Candidate.query.join(Party).add_columns(
            Candidate.candidate_name, Candidate.area,
            Party.party_name, Candidate.party_id
        ).all()

        return render_template(
            'add_party.html',
            all_candidates=all_candidates,
            solapur_talukas=solapur_talukas,
            maharashtra_parties=maharashtra_parties,
            parties_logo_map=parties_logo_map,
            message=locals().get('message'),
            message_type=locals().get('message_type')
        )

    return redirect(url_for('login'))


@app.route('/schedule_election', methods=['GET', 'POST'])
def schedule_election():
    if request.method == 'POST':
        election_date = request.form.get('election_date')
        start_time = request.form.get('start_time')
        end_time = request.form.get('end_time')

        # Validate that all fields are provided
        if not (election_date and start_time and end_time):
            flash('All fields are required', 'warning')
            return redirect(url_for('schedule_election'))

        # Convert input date and time to proper datetime objects
        election_date = datetime.strptime(election_date, '%Y-%m-%d').date()
        start_time = datetime.strptime(start_time, '%H:%M').time()
        end_time = datetime.strptime(end_time, '%H:%M').time()

        # Get current time and ensure election time is in the future
        now = datetime.now()
        current_date = now.date()
        current_time = now.time()

        if election_date < current_date or (election_date == current_date and start_time <= current_time):
            flash("Election time must be in the future.", "danger")
            return redirect(url_for('schedule_election'))

        # Check if the new election time overlaps with any existing elections on the same date
        overlapping_elections = Election.query.filter(
            Election.date == election_date,
            or_(
                and_(
                    Election.start_time <= start_time,
                    Election.end_time > start_time  # Overlapping with the new election's start time
                ),
                and_(
                    Election.start_time < end_time,
                    Election.end_time >= end_time  # Overlapping with the new election's end time
                ),
                and_(
                    Election.start_time >= start_time,
                    Election.end_time <= end_time  # Entire election within the new election time range
                ),
                and_(
                    Election.start_time <= start_time,
                    Election.end_time >= end_time  # New election is completely within an existing one
                )
            )
        ).all()

        if overlapping_elections:
            flash("Cannot schedule election, the selected time overlaps with another election.", "danger")
            return redirect(url_for('schedule_election'))

        # If no overlap, proceed to schedule the new election
        new_election = Election(
            date=election_date,
            start_time=start_time,
            end_time=end_time
        )
        db.session.add(new_election)
        db.session.commit()
        flash("Election scheduled successfully!", "success")
        return redirect(url_for('schedule_election'))

    return render_template('schedule_election.html')



@app.route('/cast_vote', methods=['GET', 'POST'])
def cast_vote():
    print("cast voter")

    # Check if the user is logged in and is a voter
    if 'userid' not in session or session['role'] != 'voter':
        flash("Unauthorized access.", "danger")
        return redirect(url_for('login'))

    if request.method == 'POST':
        # Get form data
        voter_id = request.form.get('voter_id')
        party_name = request.form.get('party_name')
        area = request.form.get('area')

        # Get today's election from DB
        today = date.today()
        now = datetime.now().time()
        election = Election.query.filter_by(date=today).filter(
            Election.start_time <= now, Election.end_time >= now
        ).first()

        election_id = election.id if election else None

        print("Voter ID:", voter_id)
        print("Party Name:", party_name)
        print("Election ID:", election_id)
        print("Area:", area)

        # Validate form data
        if not voter_id or not party_name or not election_id:
            flash("Missing required information.", "danger")
            return redirect(url_for('voter_dashboard'))

        # Check if the user already voted
        if blockchain.has_voted(voter_id, election_id):
            flash("You have already voted in this election.", "warning")
            return redirect(url_for('voter_dashboard'))

        # Record vote in blockchain
        tx_index = blockchain.new_transaction(voter_id, party_name, election_id, area)
        last_block = blockchain.last_block
        proof = blockchain.proof_of_work(last_block['proof'])
        previous_hash = blockchain.hash(last_block)
        new_block = blockchain.new_block(proof, previous_hash)

        # Log block info
        block_hash = blockchain.hash(new_block)
        timestamp = datetime.utcfromtimestamp(new_block['timestamp'])
        app.logger.info(f"--- New Vote Cast ---")
        app.logger.info(f"Voter ID     : {voter_id}")
        app.logger.info(f"Election ID  : {election_id}")
        app.logger.info(f"Party Chosen : {party_name}")
        app.logger.info(f"Block Index  : {new_block['index']}")
        app.logger.info(f"Block Hash   : {block_hash}")
        app.logger.info(f"Timestamp    : {timestamp}")
        app.logger.info(f"Transactions : {json.dumps(new_block['transactions'], indent=4)}")
        app.logger.info("----------------------")

        # Clear session
        session.pop('userid', None)
        session.pop('role', None)

        flash("Your vote has been securely cast and recorded on the blockchain!", "success")
        return redirect(url_for('home'))

    # For GET requests, render the dashboard (optional fallback)
    return redirect(url_for('voter_dashboard'))


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

@app.route('/admin_dashboard')
def admin_dashboard():
    if 'userid' in session and session['role'] == 'admin':
        return render_template('admin_dashboard.html')
    return redirect(url_for('login'))

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

    # Check if there are no votes
    if not votes:
        return render_template('view_votes.html', message="No votes found.")

    # Dictionary to store vote counts by party, election, and area
    vote_counts = {}

    # Process each vote and count the number of votes per party and election
    for vote in votes:
        election = vote.get('election_id')  # Assuming each vote has an election identifier
        party = vote.get('party_name')
        area = vote.get('area')  # Assuming area information is present

        if election not in vote_counts:
            vote_counts[election] = {}

        if party not in vote_counts[election]:
            vote_counts[election][party] = {'count': 0, 'areas': {}}

        if area not in vote_counts[election][party]['areas']:
            vote_counts[election][party]['areas'][area] = 0

        # Increment vote count for the specific area
        vote_counts[election][party]['areas'][area] += 1
        vote_counts[election][party]['count'] += 1

    return render_template('view_votes.html', vote_counts=vote_counts)


# Ensure tables are created when the application starts
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)
