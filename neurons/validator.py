# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Bittensor Validator Template:
# TODO(developer): Rewrite based on protocol defintion.

# Step 1: Import necessary libraries and modules
import os
import time
import torch
import argparse
import traceback
import bittensor as bt

# import this repo
import template

from .misc import TrainingModel

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np

from transformers import ViTForImageClassification




# Step 2: Set up the configuration parser
# This function is responsible for setting up and parsing command-line arguments.
def get_config():

    parser = argparse.ArgumentParser()
    # TODO(developer): Adds your custom validator arguments to the parser.
    parser.add_argument('--custom', default='my_custom_value', help='Adds a custom value to the parser.')
    # Adds override arguments for network and netuid.
    parser.add_argument( '--netuid', type = int, default = 1, help = "The chain subnet uid." )
    # Adds subtensor specific arguments i.e. --subtensor.chain_endpoint ... --subtensor.network ...
    bt.subtensor.add_args(parser)
    # Adds logging specific arguments i.e. --logging.debug ..., --logging.trace .. or --logging.logging_dir ...
    bt.logging.add_args(parser)
    # Adds wallet specific arguments i.e. --wallet.name ..., --wallet.hotkey ./. or --wallet.path ...
    bt.wallet.add_args(parser)
    # Parse the config (will take command-line arguments if provided)
    # To print help message, run python3 template/miner.py --help
    config =  bt.config(parser)

    # Step 3: Set up logging directory
    # Logging is crucial for monitoring and debugging purposes.
    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            'validator',
        )
    )
    # Ensure the logging directory exists.
    if not os.path.exists(config.full_path): os.makedirs(config.full_path, exist_ok=True)

    # Return the parsed config.
    return config

validator_model = TrainingModel(validator_model=ViTForImageClassification.from_pretrained('againeureka/vit_cifar10_classification'))

def secret_evaluation_task():
    """
    Evaluate the provided validator_model on the CIFAR-10 dataset.

    Parameters:
    - validator_model : torch.nn.Module - The validator_model to be evaluated.
    - batch_size : int - The size of the batches used for evaluation.
    - num_classes : int - Number of classes in the classification task (default is 10 for CIFAR-10)
    - use_cuda : bool - Use CUDA if it's available

    Returns:
    - float : accuracy of the validator_model on the CIFAR-10 test dataset.
    """
    batch_size=32
    num_classes=10
    use_cuda=False
    # Check CUDA availability and set the device accordingly
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    validator_model.to(device)
    
    # Define a transformation that you apply to the CIFAR-10 test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizing using mean and std of CIFAR-10
    ])
    
    # Load the CIFAR-10 test dataset
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Set the validator_model to evaluation mode
    validator_model.eval()
    
    # Initialize a confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    # Define the loss criterion - using Cross Entropy Loss for multi-class classification
    criterion = nn.CrossEntropyLoss()
    
    # Initialize variables to keep track of loss and correct predictions
    total_loss = 0.0
    total_correct = 0
    
    # Loop through the test data
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = validator_model(inputs)
            
            # Compute the loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Get the predicted classes
            _, predicted = torch.max(outputs, 1)
            
            # Update the confusion matrix
            for i in range(batch_size):
                confusion_matrix[labels[i], predicted[i]] += 1
                
            # Update the correct predictions counter
            total_correct += (predicted == labels).sum().item()
    
    # Compute accuracy
    accuracy = 100 * total_correct / len(testset)
    
    # You might also want to return other metrics, such as precision, recall, F1 score, etc., which can be computed from the confusion matrix
    # For now, let's just return the accuracy
    return accuracy    

def evaluate_delta_on_metrics(delta):
    
    ## Update validator_model with delta
    validator_model.apply_delta(delta) 

    ## Evaluate validator_model performance on task
    model_score = secret_evaluation_task()

    return model_score

def assert_weight_concensus(new_weight, metagraph):
    raise NotImplementedError
    renegades = []
    for validator_value,validator_uid in validator_values: #FIXME psuedocode
        try:
            assert new_weight == validator_value
        except AssertionError:
            renegades.append(validator_uid)
    if len(renegades) == 0:
        return []
    else:
        return renegades
    #ensures all validators agree on a weight, if not can blacklist renegade validators after timeout period has passed
    #How would it be possible to read back all the blocks if joining later? 
    #Or should it be state agnostic and just send compressed weights and call it a day?
    #Should there be an oracle? that observes and gives latest directions
    #No best way is to maintain that all weights are compressed deltas from the initial weights not the intermediate weight
    #this will save the memory needed and since their is no dependance it works (bad for rollbacks though)
    
def agree_with_concensus(renegades):
    raise NotImplementedError

def blacklist_validators(renegades):
    raise NotImplementedError
    # Can choose to blacklist straightaway or apply different penalties/tardiness scores

def get_new_delta(initial_weights, new_delta):
    """
    Computes the new delta (difference) between the initial weights and new weights (delta).

    Parameters:
        initial_weights (torch.Tensor): The initial weights as a concatenated tensor.
        new_delta (torch.Tensor): The new weights or delta as a concatenated tensor.

    Returns:
        torch.Tensor: The calculated delta to be applied to the model weights.
    """
    # Ensure that initial_weights and new_delta have the same number of elements
    assert initial_weights.numel() == new_delta.numel(), "Mismatch in number of elements between initial weights and new delta"

    # Calculate the new delta: difference between the new weights (delta) and initial weights
    computed_delta = new_delta - initial_weights

    return computed_delta

## FIXME what ensures that validators don't become lazy and just echo each other
def set_new_delta_input(metagraph):
    ## Get the best performing weight according to the last pass
    ## TODO add weight averaging 
    import pdb;pdb.set_trace()
    new_delta = metagraph.axons[max(metagraph.weights)] #FIXME check syntax, 
    #FIXME gets axon not their weight

    ## The validators need to be done with the assesment in an agreed upon waiting period
    # time_begin = time()
    # while time() - time_begin < waiting_period:
    #     renegades = assert_weight_concensus(new_weight)

    # #if there are any renegade validators blacklist them
    # blacklist_validators(renegades)

    # #check self is not renegade
    # agree_with_concensus(renegades)

    return get_new_delta(validator_model.concatenated_initial_weights, new_delta)
    #The goal is to get the weights of the miner response with the highest value
    #FIXME. prone to race conditions

### Each validator can come up with their own method of validator_model evaluation
def model_evaluator(validator_model, weight):
    raise NotImplementedError
    ## Add a simple CIFAR-10 evaluator here

#TODO add typing Synapse --> Synapse
def evaluate_response_delta(response_delta, model_evaluator):
    reconstructed_weight = reconstruct_delta(initial_weights, response_delta)
    score = model_evaluator(validator_model, weight)
    return score

def main( config ):
    # Set up logging with the provided configuration and directory.
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(f"Running validator for subnet: {config.netuid} on network: {config.subtensor.chain_endpoint} with config:")
    # Log the configuration for reference.
    bt.logging.info(config)

    # Step 4: Build Bittensor validator objects
    # These are core Bittensor classes to interact with the network.
    bt.logging.info("Setting up bittensor objects.")

    # The wallet holds the cryptographic key pairs for the validator.
    wallet = bt.wallet( config = config )
    bt.logging.info(f"Wallet: {wallet}")

    # The subtensor is our connection to the Bittensor blockchain.
    subtensor = bt.subtensor( config = config )
    bt.logging.info(f"Subtensor: {subtensor}")

    # Dendrite is the RPC client; it lets us send messages to other nodes (axons) in the network.
    dendrite = bt.dendrite( wallet = wallet )
    bt.logging.info(f"Dendrite: {dendrite}")

    # The metagraph holds the state of the network, letting us know about other miners.
    metagraph = subtensor.metagraph( config.netuid )
    bt.logging.info(f"Metagraph: {metagraph}")

    # Step 5: Connect the validator to the network
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        bt.logging.error(f"\nYour validator: {wallet} if not registered to chain connection: {subtensor} \nRun btcli register and try again.")
        exit()
    else:
        # Each miner gets a unique identity (UID) in the network for differentiation.
        my_subnet_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
        bt.logging.info(f"Running validator on uid: {my_subnet_uid}")

    # Step 6: Set up initial scoring weights for validation
    bt.logging.info("Building validation weights.")
    alpha = 0.9
    scores = torch.ones_like(metagraph.S, dtype=torch.float32)
    bt.logging.info(f"Weights: {scores}")

    # Step 7: The Main Validation Loop
    bt.logging.info("Starting validator loop.")
    step = 0
    while True:
        try:
            # TODO(developer): Define how the validator selects a miner to query, how often, etc.
            # Broadcast a query to all miners on the network.
            responses = dendrite.query(
                # Send the query to all axons in the network.
                metagraph.axons,
                # Construct a dummy query.
                template.protocol.Weight( delta_input = set_new_delta_input(metagraph) ), # Construct a dummy query.
                # All responses have the deserialize function called on them before returning.
                deserialize = True, 
            )

            # Log the results for monitoring purposes.
            bt.logging.info(f"Received dummy responses: {responses}")

            # TODO(developer): Define how the validator scores responses.
            # Adjust the scores based on responses from miners.
            for i, resp_i in enumerate(responses):
                # Initialize the score for the current miner's response.
                score = 0

                # Check if the miner has provided the correct response by doubling the dummy input.
                # If correct, set their score for this round to 1.
                # if resp_i == step * 2:
                #     score = 1
                score = evaluate_delta_on_metrics(resp_i)

                # Update the global score of the miner.
                # This score contributes to the miner's weight in the network.
                # A higher weight means that the miner has been consistently responding correctly.
                scores[i] = alpha * scores[i] + (1 - alpha) * score

            # Periodically update the weights on the Bittensor blockchain.
            if (step + 1) % 2 == 0:
                # TODO(developer): Define how the validator normalizes scores before setting weights.
                weights = torch.nn.functional.normalize(scores, p=1.0, dim=0)
                bt.logging.info(f"Setting weights: {weights}")
                # This is a crucial step that updates the incentive mechanism on the Bittensor blockchain.
                # Miners with higher scores (or weights) receive a larger share of TAO rewards on this subnet.
                result = subtensor.set_weights(
                    netuid = config.netuid, # Subnet to set weights on.
                    wallet = wallet, # Wallet to sign set weights using hotkey.
                    uids = metagraph.uids, # Uids of the miners to set weights for.
                    weights = weights, # Weights to set for the miners.
                    wait_for_inclusion = True
                )
                if result: bt.logging.success('Successfully set weights.')
                else: bt.logging.error('Failed to set weights.') 

            # End the current step and prepare for the next iteration.
            step += 1
            # Resync our local state with the latest state from the blockchain.
            metagraph = subtensor.metagraph(config.netuid)
            # Sleep for a duration equivalent to the block time (i.e., time between successive blocks).
            time.sleep(bt.__blocktime__)

        # If we encounter an unexpected error, log it for debugging.
        except RuntimeError as e:
            bt.logging.error(e)
            traceback.print_exc()

        # If the user interrupts the program, gracefully exit.
        except KeyboardInterrupt:
            bt.logging.success("Keyboard interrupt detected. Exiting validator.")
            exit()

# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    # Parse the configuration.
    config = get_config()
    # Run the main function.
    main( config )
