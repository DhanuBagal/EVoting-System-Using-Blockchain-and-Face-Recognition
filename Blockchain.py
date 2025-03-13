import hashlib
import json
from time import time

class Blockchain:
    def __init__(self):
        self.current_transactions = []
        self.chain = []
        self.voter_votes = {}  # To track if a voter has already voted
        self.new_block(previous_hash='1', proof=100)  # Create the genesis block

    def new_transaction(self, voter_id, party_name):
        """Creates a new transaction (vote) to go into the next mined block"""
        # Check if the voter has already voted
        if voter_id in self.voter_votes:
            return f"Voter {voter_id} has already voted for {self.voter_votes[voter_id]}."

        # Create a new transaction
        transaction = {
            'voter_id': voter_id,
            'party_name': party_name,
            'timestamp': time()  # Add timestamp to the vote
        }
        self.current_transactions.append(transaction)

        # Mark this voter as having voted
        self.voter_votes[voter_id] = party_name

        # Return the index of the transaction
        return len(self.current_transactions) - 1

    def new_block(self, proof, previous_hash=None):
        """Creates a new block and adds it to the chain"""
        block = {
            'index': len(self.chain) + 1,
            'timestamp': time(),
            'transactions': self.current_transactions,  # Add current transactions to the block
            'proof': proof,
            'previous_hash': previous_hash or self.hash(self.chain[-1]),
        }
        self.current_transactions = []  # Reset the current transactions list after mining the block
        self.chain.append(block)
        return block

    def proof_of_work(self, last_proof):
        """Proof of work algorithm (simple POW)"""
        proof = 0
        while self.valid_proof(last_proof, proof) is False:
            proof += 1
        return proof

    @staticmethod
    def valid_proof(last_proof, proof):
        """Validates the proof: Does hash(last_proof, proof) contain 4 leading zeroes?"""
        guess = f'{last_proof}{proof}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:4] == "0000"

    @staticmethod
    def hash(block):
        """Creates a SHA-256 hash of a block"""
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    @property
    def last_block(self):
        """Returns the last block in the chain"""
        return self.chain[-1]

    def get_votes(self):
        """Returns a list of all votes cast, including the block hash"""
        votes = []
        for block in self.chain:
            block_hash = self.hash(block)  # Compute the block hash
            for transaction in block['transactions']:
                # Add block hash to each transaction (vote)
                vote_with_hash = transaction.copy()  # Copy the transaction to avoid modifying the original
                vote_with_hash['block_hash'] = block_hash
                vote_with_hash['timestamp'] = block['timestamp']  # Include the block's timestamp
                votes.append(vote_with_hash)
        return votes
