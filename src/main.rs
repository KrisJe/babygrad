
mod engine;
mod nn;

use engine::Value;
use nn::Neuron;


fn main() {

    let a = Value::new(-2.0);
    let b = Value::new(3.0);
    
    let c = a.clone() + b.clone();
    //let c = a.clone() / b.clone();
    //let c = a.clone() * b.clone();
    //let c = a.clone() - b.clone();

    println!("Value[c]={}", c.value());
    println!("Gradient[c]={}", c.gradient());
}

