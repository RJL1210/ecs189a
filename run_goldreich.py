# main.py or run.py
from goldreich import goldreich_reduction
from goldreich_tester import batch_test
from visualization import visualize_batch_results

if __name__ == "__main__":
    test_dir = './D'
    sample_dir = './X'
    epsilon = 0.1
    
    # Run batch tests
    results = batch_test(test_dir, sample_dir, epsilon)
    
    # Visualize results
    #WILL GENERATE LOT OF IMAGES
    
    #visualize_batch_results(test_dir, sample_dir, epsilon)