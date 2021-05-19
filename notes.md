# Week 7 Notes

## Neural Networks: Backpropagation

We've already presented inputs to the neural 
network at the backpropagation phase

- Backpropogation takes output values
and compares them to the expected value
  
- Expected value is 1–true, 0-false

- We use delta to figure out how far off our prediction is

**Error** = presented (input) value - expected value

After we have the **error**, we propegate it back
through the network from the output layer to the
input layer

#### to do this, we use gradient descent
each step, we look at input nodes and make tiny adjustments
to try and make our error towards 0

## Neural Networks: Backpropagation Workflow
Starts: output module uses fire_upstream to indicate
to neurode that it's ready

gets info that it's ready
checks to see that all are ready
if not ready, wait

1) wait for downstream nodes
2) calculate delta
3) Fire upstream
4) update downstream weights