from flask import render_template, request, redirect, url_for
from app import app
from models import db, User, Vote
from Blockchain import add_vote_to_blockchain

@app.route('/add_party', methods=['GET', 'POST'])
def add_party():
    if request.method == 'POST':
        # Logic to add a new party to the database
        party_name = request.form['party_name']
        # Add party logic here
        return redirect(url_for('admin_dashboard'))
    return render_template('add_party.html')

@app.route('/view_votes')
def view_votes():
    # Logic to view all votes
    votes = Vote.query.all()
    return render_template('view_votes.html', votes=votes)

@app.route('/schedule_election', methods=['GET', 'POST'])
def schedule_election():
    if request.method == 'POST':
        # Logic to schedule an election
        election_date = request.form['election_date']
        # Schedule election logic here
        return redirect(url_for('admin_dashboard'))
    return render_template('schedule_election.html')

@app.route('/cast_vote', methods=['POST'])
def cast_vote():
    user_id = session['userid']
    candidate = request.form['candidate']
    vote_data = {
        'user_id': user_id,
        'candidate': candidate
    }
    block = add_vote_to_blockchain(vote_data)
    new_vote = Vote(user_id=user_id, candidate=candidate, blockchain_hash=block['hash'])
    db.session.add(new_vote)
    db.session.commit()
    return "Vote cast successfully!"