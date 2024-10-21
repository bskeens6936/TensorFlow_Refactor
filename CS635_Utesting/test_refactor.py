'''
    Brandon Skeens
    826416091
    To run unit tests, run:
        python3 -m unittest test_refactor.py 
        (not sure if this changes between machines, I have python3 on mine so the cmd might be slightly different)
    Will test the added sub functions to ensure the behavior is expected
'''

import unittest
# Importing individually, not sure why it was freaking out but
# this was the solution I found
import tensorflow.python.keras as tfKeras
import tensorflow.python.keras.layers as tfLayers

# Import where refactor was done
from tensorflow.python.keras.engine import training

# Define unit tests
class TestYourModelClass(unittest.TestCase):
    def test_default_initialization(self):
        model = training.Model()
        self.assertIsNone(model.compiled_loss)
        self.assertIsNone(model.compiled_metrics)

    def test_subclassed_functional_model_initialization(self):
        inputs = tfKeras.Input(shape=(32,))
        outputs = tfLayers.Dense(1)(inputs)
        model = training.Model(inputs=inputs, outputs=outputs)
        self.assertTrue(model._is_model_for_instrumentation)
        self.assertIsNotNone(model.inputs)
        self.assertIsNotNone(model.outputs)

    def test_invalid_keyword_arguments(self):
        with self.assertRaises(TypeError):
            training.Model(invalid_param='value')

    def test_attribute_initialization(self):
        model = training.Model()
        self.assertEqual(model.stop_training, False)
        self.assertIsNone(model.history)
     
    def test_model_state_after_initialization(self):
        model = training.Model()
        self.assertFalse(model._is_graph_network)

    def test_base_model_initialized(self):
        model = training.Model()
        self.assertTrue(model._base_model_initialized)

# Run unit tests
if __name__ == '__main__':
    unittest.main()
