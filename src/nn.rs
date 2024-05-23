use rand::distributions::{Distribution, Uniform};
use std::iter::zip;
//mod engine;


//use engine::Value;
use crate::Value;
/*
import random
from micrograd.engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []
*/

/* 
class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"
*/

#[derive(Debug)]
pub enum ActivationFunc {
    None,
    Relu,
    Tanh,
    Linear,
    Sigmoid,
    Gelu
}


#[derive(Debug)]
pub struct Neuron {
    pub weights: Vec<Value>,
    pub bias: Value,
    pub nonlin: bool,
}

impl Neuron {
    pub fn new(w_input_size: usize, nonlin: bool) -> Neuron {   
        let mut rng = rand::thread_rng();
        let uniform = Uniform::new_inclusive(-1.0, 1.0);
        
        let mut weights: Vec<Value> = Vec::new();
        for _ in 1..=w_input_size {
            weights.push(Value::new(uniform.sample(&mut rng)));
        }

        Neuron{ 
            weights : weights,
            bias: Value::new(0.0), //Value::new(uniform.sample(&mut rng)),
            nonlin : nonlin,           
        }
    }

    pub fn call(&self, inputs: Vec<Value>, act: ActivationFunc)-> Value{        
        let mut output = self.bias.clone();
        for (input, weight) in zip(inputs, self.weights.iter()) {
            output = output + input * weight.clone();
        }       
        if self.nonlin == false {
            return output;
        }
        else{
            match act {  
                //ActivationFunc::Relu => println!("Applying ReLU activation function"),            
                //ActivationFunc::Linear => println!("Applying Linear activation function"),
                //ActivationFunc::Sigmoid => println!("Applying Sigmoid activation function"),
                //ActivationFunc::Gelu => println!("Applying GELU activation function"),          
                ActivationFunc::Tanh => output.tanh(),
                _ => output,          
            }   
        }   
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut result = self.weights.clone();
        result.push(self.bias.clone());
        result
    }
}


/*
class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
*/



#[derive(Debug)]
pub struct MLP {
    pub layers: Vec<Neuron>,
    pub act: ActivationFunc,
}

/* 
class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
        
*/





#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_neutron() {
        let lin = Neuron::new(3, true);
        assert_eq!(lin.parameters().len(), 4);    
    }



}