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
#[allow(dead_code)]
#[derive(Debug, Clone)]
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

    pub fn forward(&self, inputs: Vec<Value>, act: ActivationFunc)-> Value{        
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
pub struct Layer {
    pub neurons: Vec<Neuron>,
}

#[allow(dead_code)]
impl Layer {
    fn new(input_size: usize, output_size: usize) -> Layer {
        let mut neurons: Vec<Neuron> = Vec::new();
        for _ in 1..=output_size {
            neurons.push(Neuron::new(input_size, false));
        }
        Layer { neurons}
    }

    fn forward(&self, inputs: Vec<Value>, act: ActivationFunc) -> Vec<Value> {
        let mut result: Vec<Value> = Vec::new();
        for neuron in self.neurons.iter() {
            result.push(neuron.forward(inputs.clone(), act.clone()));
        }
        result
    }

    fn parameters(&self) -> Vec<Value> {
        let mut parameters: Vec<Value> = Vec::new();
        for neuron in self.neurons.iter() {
            parameters.append(&mut neuron.parameters())
        }
        parameters
    }
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


#[derive(Debug)]
pub struct MLP {
    pub layers: Vec<Layer>,
    //pub act: ActivationFunc,
}


#[allow(dead_code)]
impl MLP {
    fn new(input_size: usize, hidden_layers_size: &[usize]) -> MLP {
        let mut layers: Vec<Layer> = Vec::new();
        let hlc = hidden_layers_size.len();
        layers.push(Layer::new(input_size, hidden_layers_size[0]));
        for i in 0..hlc - 1 {
            layers.push(Layer::new(hidden_layers_size[i], hidden_layers_size[i + 1]))
        }
        layers.push(Layer::new(hidden_layers_size[hlc - 1], 1));
        MLP {layers}
    }

    fn forward(&self, inputs: Vec<Value>) -> Value {
        let mut outputs: Vec<Value> = inputs;
        for layer in self.layers.iter() {
            outputs = layer.forward(outputs, ActivationFunc::None)
        }
        assert!(outputs.len() == 1);
        outputs[0].clone()
    }

    fn shape(&self) -> Vec<usize> {
        let mut sizes: Vec<usize> = Vec::new();
        sizes.push(self.layers[0].neurons[0].weights.len());
        sizes.append(&mut self.layers.iter().map(|el| el.neurons.len()).collect::<Vec<usize>>());
        sizes
    }

    fn parameters(&self) -> Vec<Value> {
        let mut parameters: Vec<Value> = Vec::new();
        for layer in self.layers.iter() {
            parameters.append(&mut layer.parameters())
        }
        parameters
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_neutron() {
        let lin = Neuron::new(3, false);
        assert_eq!(lin.parameters().len(), 4);    
    }

    #[test]
    fn layer() {   

        let layer1 = Layer::new(4,4);
        let mut inputs: Vec<Value> = Vec::new();
        layer1.forward(inputs, ActivationFunc::None);
        assert_eq!(layer1.parameters().len(), 20); 
    }


    #[test]
    fn MLP() {       
        let xs: &[&[f64]] = &[
            &[1.0, 6.0, 0.0],
            &[0.0, 3.0, 1.0],
            &[2.0, 4.0, 0.0],
            &[0.0, 3.0, 2.0],
            &[3.0, 2.0, 0.0],
            &[0.0, 1.0, 3.0],
        ];  
       
        let mlp = MLP::new(3, &[4, 4]);
        assert_eq!(mlp.parameters().len(), 41); 
        let output = mlp.forward(Value::vec(xs[0]));
  
        assert_eq!(mlp.parameters().len(), 41);
    }



}