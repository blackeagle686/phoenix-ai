from phoenix.memory.utils.scoring import calculate_recency_score, calculate_hybrid_score
from helpers import print_step
import time

def test_scoring_utils():
    print_step("Calculating recency score for now")
    now = time.time()
    score_now = calculate_recency_score(now)
    assert score_now > 0.99
    
    print_step("Calculating recency score for 1000s ago")
    then = now - 1000
    score_then = calculate_recency_score(then)
    assert score_then < score_now
    
    print_step("Calculating hybrid score")
    hybrid = calculate_hybrid_score(relevance=0.8, importance=0.7, recency=0.9)
    assert 0.7 < hybrid < 0.9
    
    print_step("Scoring utils validation passed")

if __name__ == "__main__":
    test_scoring_utils()

