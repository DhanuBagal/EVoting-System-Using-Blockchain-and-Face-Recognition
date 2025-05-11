import hashlib
import json
from time import time
import os

class Blockchain:
    def __init__(self, filename="blockchain.json"):
        self.current_transactions = []
        self.chain = []
        self.voter_votes = {}  # election_id -> {voter_id: (party_name, area)}
        self.filename = filename

        # Load the blockchain from file if it exists
        if os.path.exists(self.filename):
            self.load_chain()
        else:
            # If no blockchain file exists, create a new blockchain
            self.new_block(previous_hash='1', proof=100)

    def new_transaction(self, voter_id, party_name, election_id, area):
        if election_id not in self.voter_votes:
            self.voter_votes[election_id] = {}

        # Check if the voter has already voted in this election
        if voter_id in self.voter_votes[election_id]:
            return f"Voter {voter_id} has already voted for {self.voter_votes[election_id][voter_id][0]} in election {election_id}."

        # Create the vote transaction
        transaction = {
            'election_id': election_id,
            'voter_id': voter_id,
            'party_name': party_name,
            'area': area,
            'timestamp': time()
        }
        self.current_transactions.append(transaction)

        # Mark the voter as voted in this election, storing the party and area
        self.voter_votes[election_id][voter_id] = (party_name, area)

        # Save the blockchain after the new transaction
        self.save_chain()

        return len(self.chain) + 1  # index of the block that will hold this transaction

    def new_block(self, proof, previous_hash=None):
        block = {
            'index': len(self.chain) + 1,
            'timestamp': time(),
            'transactions': self.current_transactions,
            'proof': proof,
            'previous_hash': previous_hash or self.hash(self.chain[-1]),
        }
        self.current_transactions = []  # reset the transaction list
        self.chain.append(block)

        # Save the blockchain after adding the new block
        self.save_chain()

        return block

    def proof_of_work(self, last_proof):
        proof = 0
        while not self.valid_proof(last_proof, proof):
            proof += 1
        return proof

    @staticmethod
    def valid_proof(last_proof, proof):
        guess = f'{last_proof}{proof}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:4] == "0000"

    @staticmethod
    def hash(block):
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    @property
    def last_block(self):
        return self.chain[-1]

    def get_votes(self):
        votes = []
        for block in self.chain:
            block_hash = self.hash(block)
            for tx in block['transactions']:
                vote = tx.copy()
                vote['block_hash'] = block_hash
                vote['block_timestamp'] = block['timestamp']
                votes.append(vote)
        return votes

    def tally_votes(self, election_id):
        results = {}
        for block in self.chain:
            for tx in block['transactions']:
                if tx['election_id'] == election_id:
                    party = tx['party_name']
                    area = tx['area']
                    if party not in results:
                        results[party] = {}

                    if area not in results[party]:
                        results[party][area] = 0

                    results[party][area] += 1

        return results

    def has_voted(self, voter_id, election_id):
        return voter_id in self.voter_votes.get(election_id, {})

    def save_chain(self):
        """
        Saves the blockchain to a file in JSON format
        """
        with open(self.filename, 'w') as file:
            json.dump({
                'chain': self.chain,
                'current_transactions': self.current_transactions,
                'voter_votes': self.voter_votes
            }, file, indent=4)

    def load_chain(self):
        """
        Loads the blockchain from a file
        """
        with open(self.filename, 'r') as file:
            data = json.load(file)
            self.chain = data['chain']
            self.current_transactions = data['current_transactions']
            self.voter_votes = data['voter_votes']
