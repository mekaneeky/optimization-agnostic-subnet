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

import typing
import bittensor as bt
import torch
# TODO(developer): Rewrite with your protocol definition.

# This is the protocol for the dummy miner and validator.
# It is a simple request-response protocol where the validator sends a request
# to the miner, and the miner responds with a dummy response.

# ---- miner ----
# Example usage:
#   def dummy( synapse: Dummy ) -> Dummy:
#       synapse.dummy_output = synapse.dummy_input + 1
#       return synapse
#   axon = bt.axon().attach( dummy ).serve(netuid=...).start()

# ---- validator ---
# Example usage:
#   dendrite = bt.dendrite()
#   dummy_output = dendrite.query( Dummy( dummy_input = 1 ) )
#   assert dummy_output == 2

class Weight( bt.Synapse ):
    """
    A simple dummy protocol representation which uses bt.Synapse as its base.
    This protocol helps in handling dummy request and response communication between
    the miner and the validator.

    Initial initialization can be from existing model or shared seed + parameter distribution to minimize commnuication needed. 
    Then there could be a decentralized record of deltas

    Attributes:
    - delta_input: A delta value of the latest concensus weight
    - delta_output: A delta/compressed delta (TODO can validators penalize on communication size too) that is 
    tested by the validators on the hidden evaluation metrics.
    """

    # Required request input, filled by sending dendrite caller.
    delta_input: int
    init_seed: int

    # Required request input hash, filled automatically when dendrite creates the request.
    # This allows for proper data validation and messages are signed with the hashes of the
    # required body fields. Ensure you have a {field}_hash field for each required field. 
    # 
    delta_input_hash: str = ""
    init_seed_hash: str = ""
    # Optional request output, filled by recieving axon.
    delta_output: typing.Optional[int] = None

    def deserialize(self) -> int:
        """
        Deserialize the dummy output. This method retrieves the response from
        the miner in the form of dummy_output, deserializes it and returns it
        as the output of the dendrite.query() call.

        Returns:
        - int: The deserialized response, which in this case is the value of dummy_output.

        Example:
        Assuming a Dummy instance has a dummy_output value of 5:
        >>> dummy_instance = Dummy(dummy_input=4)
        >>> dummy_instance.dummy_output = 5
        >>> dummy_instance.deserialize()
        5
        """
        return self.dummy_output
