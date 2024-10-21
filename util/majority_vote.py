def majority_vote(votes):

    majority = None
    majority_count = 0

    for vote in votes:
        if majority_count == 0:
            majority = vote
        if vote == majority:
            majority_count += 1
        else:
            majority_count -= 1
            
    return majority
