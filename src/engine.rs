use std::ops;
use std::{cell::{Ref, RefCell, RefMut},
     rc::Rc,
     collections::{HashMap, HashSet},
     sync::atomic::AtomicUsize};
use std::fmt;


extern crate graphviz_rust;
use graphviz_rust::dot_structures::*;
use graphviz_rust::parse;
use graphviz_rust::dot_generator::*;
use graphviz_rust::attributes::GraphAttributes as GAttributes;
use self::graphviz_rust::attributes::{EdgeAttributes, NodeAttributes, rankdir, shape};
use self::graphviz_rust::printer::{DotPrinter, PrinterContext};


#[allow(dead_code)]
#[derive(Clone, Debug)]
pub enum Op {
    None,
    Add,
    Mul,
    Tanh,
    Exp,
    Pow,
    Sub,
    Div,
    Relu,
    Neg
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct ValueData {
    value: f64,
    children: Vec<Value>,
    gradient: f64,
    op: Op,
    visited: bool,
    id: usize, //for sorting
}

static VAL_CNT: AtomicUsize = AtomicUsize::new(0);

impl ValueData {
    
    fn new(value: f64) -> ValueData {            
        ValueData {
            value: value,
            children: vec![],
            gradient: 0.0,
            op: Op::None,
            visited: false,
            id : 0
        }
    }

   
}


#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct Value(Rc<RefCell<ValueData>>);

impl Value {
    pub fn new(value: f64) -> Value {
        let id = VAL_CNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst); 
        Value(Rc::new(RefCell::new(ValueData {
            value: value,
            children: vec![], //Vec::new(),
            gradient: 0.0,
            op: Op::None,
            visited: false,
            id,
        })))
    }

    pub fn from(value: f64, children: Vec<Value>, op: Op) -> Value {
        let id = VAL_CNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst); 
        Value(Rc::new(RefCell::new(ValueData {
            value: value,
            children: children,
            gradient: 0.0,
            op: op,
            visited: false,
            id ,
        })))
    } 

    
    pub fn inner(&self) -> Ref<ValueData> {
        (*self.0).borrow()
    }

    pub fn inner_mut(&self) -> RefMut<ValueData> {
        (*self.0).borrow_mut()
    }

    fn child(&self, index: usize) -> Value {
        self.0.borrow().children[index].clone()
    }

    fn only_child(&self) -> Value {
        assert!(self.0.borrow().children.len() == 1);
        self.child(0)
    }

    fn lhs(&self) -> Value {
        assert!(self.0.borrow().children.len() == 2);
        self.child(0)
    }
    
    
    fn rhs(&self) -> Value {
        assert!(self.0.borrow().children.len() == 2);
        self.child(1)
    }

    pub fn op(&self) -> Op {
        self.0.borrow().op.clone()
    }

    fn visited(&self) -> bool {
        self.0.borrow().visited
    }


    pub fn value(&self) -> f64 {
        self.0.borrow().value
    }

    pub fn gradient(&self) -> f64 {
        self.0.borrow().gradient
    }

    fn inc_gradient(&self, amount: f64) {
        self.0.borrow_mut().gradient += amount
    }

    fn zero_gradient(&self, amount: f64) {
        self.0.borrow_mut().gradient = 0.0
    }

    pub fn vec(values: &[f64]) -> Vec<Value> {
        values
            .iter()
            .map(|el| Value::new(*el))
            .collect::<Vec<Value>>()
    }

    pub fn id(&self) -> usize{
        self.0.borrow().id
    }


  
    pub fn tanh(self) -> Value {
        Value::from(self.value().tanh(), vec![self.clone()], Op::Tanh)
    }

 
    pub fn exp(self) -> Value {
        Value::from(self.value().exp(), vec![self.clone()], Op::Exp)
    }

    pub fn pow(self, value: f64) -> Value {
        Value::from(self.value().powf(value), vec![self.clone()], Op::Pow)
    }


    fn _reset_children_gradients_and_visited(node: &Value) {
        for children in node.0.borrow().children.iter() {
            children.0.borrow_mut().gradient = 0.0;
            children.0.borrow_mut().visited = false;
            Value::_reset_children_gradients_and_visited(children);
        }
    }

    fn _find_leaf_nodes_not_visited(node: &Value) -> Vec<Value> {
        let mut result: Vec<Value> = Vec::new();
        if !node.visited() {
            let mut count = 0;
            for child in node.0.borrow().children.iter() {
                if !child.visited() {
                    result.append(&mut Value::_find_leaf_nodes_not_visited(child));
                    count += 1;
                }
            }
            if count == 0 {
                node.0.borrow_mut().visited = true;
                result.push(node.clone())
            }
        }
        result
    }
    

    //we get a panic!
    fn topological_sort(&self) -> Vec<Value> {
        let mut parameters: Vec<Value> = Vec::new();
        loop {
            let mut leafs = Value::_find_leaf_nodes_not_visited(self);
            if leafs.len() == 0 {
                break;
            }
            println!("{}",leafs.len());
            parameters.append(&mut leafs);
        }        
        parameters.reverse();
        parameters
    }

   

    fn backward(&self) {
        
        Value::_reset_children_gradients_and_visited(self);

        // Run topological sorting to compute the ordered list of parameters        
        let mut parameters: Vec<Value> = Vec::new();
        loop {
            let mut leafs = Value::_find_leaf_nodes_not_visited(self);
            if leafs.len() == 0 {
                break;
            }
            parameters.append(&mut leafs);
        }
        parameters.reverse();

        /* 
        println!("{}", parameters.len()); 
        for p in parameters.clone(){
            println!("{}", p);
        }
        self.inner_mut().gradient = 1.0;*/

        parameters[0].0.borrow_mut().gradient = 1.0;
        

        // Fill in all the gradients in reverse topological order

        for node in parameters {
            let out_gradient = node.gradient();
            let out_value = node.value();

            match node.op() {               
                Op::Add => {
                    node.lhs().inc_gradient(out_gradient);
                    node.rhs().inc_gradient(out_gradient);
                }
                Op::Sub => {
                    node.lhs().inc_gradient(out_gradient);
                    node.rhs().inc_gradient(-out_gradient);
                }
                Op::Mul => {
                    node.lhs().inc_gradient(node.rhs().value() * out_gradient);
                    node.rhs().inc_gradient(node.lhs().value() * out_gradient);
                }
                Op::Div => {
                    let lhs_value = node.lhs().value();
                    let rhs_value = node.rhs().value();
                    node.lhs().inc_gradient(out_gradient / rhs_value);
                    node.rhs().inc_gradient(-lhs_value * out_gradient / (rhs_value * rhs_value));
                }
                Op::Neg => {
                    node.only_child().inc_gradient(-out_gradient);
                }
                Op::Tanh => {
                    node.only_child()
                        .inc_gradient((1.0 - out_value * out_value) * out_gradient);
                }
                Op::Exp => {
                    node.only_child().inc_gradient(out_value * out_gradient);
                }
                Op::Pow => {
                    let child_value = node.only_child().value();
                    let exponent = out_value.log10() / child_value.log10();
                    node.only_child()
                        .inc_gradient(exponent * out_value / child_value * out_gradient);
                }
                _ => (),
            }
        }
    }

    


     // Create a GraphViz dot format string representation of the graph.
     pub fn export_graph(&self) -> String {    
        fn inner(node: &Value) -> String {
            let mut opstr : String = String::new();
            let mut color : u16 = 0;
            match node.op(){               
                Op::Add  => { opstr ="add".to_owned(); color = 1; }
                Op::Sub  => { opstr ="sub".to_owned(); color = 1; }
                Op::Mul => {  opstr = "mul".to_owned(); color = 2;}
                Op::Div => {  opstr= "div".to_owned(); color = 2; }
                Op::Neg => {  opstr= "neg".to_owned(); color = 2; }
                Op::Tanh => { opstr = "tanh".to_owned(); color = 3;}
                Op::Exp => {  opstr= "exp".to_owned(); color = 4;  }
                Op::Pow => {  opstr = "pow".to_owned(); color = 5; }
                Op::Relu => { opstr = "relu".to_owned(); color = 6;}  
                Op::None => { opstr = "none".to_owned(); color = 0;}        
                _ => (),
            }
            let id = node.id();      
            let mut s = format!(
                "{} [label=\"{{{} {:.2} | {:.2}}}\", color={}];\n",
                id,
                opstr,
                node.value(),
                node.gradient(),               
                color,
            );
            //s.push_str(&format!("{}\n",node.inner().children.len()));
            for prev in node.inner().children.iter() {
                s.push_str(&inner(&prev));
                s.push_str(&format!("{} -- {};\n", id, prev.inner().id));
            }
            s
        }
    
        let mut s = format!("strict graph {{\n");
        s.push_str("rankdir=RL;\n");
        s.push_str("node [shape=record,colorscheme=set28];\n");
        s.push_str(&inner(&self));
        s.push_str("}\n");
        s
    }
}



impl ops::Add<Value> for Value {
    type Output = Value;

    fn add(self, rhs: Value) -> Value {
        Value::from(
            self.value() + rhs.value(),
            vec![self.clone(), rhs.clone()],
            Op::Add,
        )
    }
}

impl ops::Mul<Value> for Value {
    type Output = Value;

    fn mul(self, rhs: Value) -> Value {
        Value::from(
            self.value() * rhs.value(),
            vec![self.clone(), rhs.clone()],
            Op::Mul,
        )
    }
}

impl ops::Div<Value> for Value {
    type Output = Value;

    fn div(self, rhs: Value) -> Value {
        Value::from(
            self.value() / rhs.value(),
            vec![self.clone(), rhs.clone()],
            Op::Div,
        )
    }
}
/* 
impl ops::Sub<Value> for Value {
    type Output = Value;

    fn sub(self, rhs: Value) -> Value {
        self + (rhs * -1.0)
    }
}
*/
impl ops::Sub<Value> for Value {
    type Output = Value;

    fn sub(self, rhs: Value) -> Value {
        Value::from(
            self.value() + (-1.0 * rhs.value()),
            vec![self.clone(), rhs.clone()],
            Op::Sub,
        )
    }
}

impl ops::Neg for Value {
    type Output = Value;

    fn neg(self) -> Value {
        Value::new(- self.value() )        
    }
}

impl ops::Mul<Value> for f64 { //creates an issue, panic when sorting!
    type Output = Value;
    
    fn mul(self, rhs: Value) -> Value {
        Value::new(self * rhs.value())       
    }
} 

impl ops::Add<Value> for f64 {
    type Output = Value;
    
    fn add(self, rhs: Value) -> Value {
        Value::new(self + rhs.value())
    }
}

impl ops::Div<Value> for f64 {
    type Output = Value;
    
    fn div(self, rhs: Value) -> Value {
        Value::new(self / rhs.value())
    }
} 

impl ops::Sub<Value> for f64 {
    type Output = Value;
    
    fn sub(self, rhs: Value) -> Value {
        Value::new(self +(-1.0 * rhs.value()))
    }
} 

impl ops::Add<f64> for Value {
    type Output = Value;

    fn add(self, rhs: f64) -> Value {
        self + Value::new(rhs)
    }
}

impl ops::Sub<f64> for Value {
    type Output = Value;

    fn sub(self, rhs: f64) -> Value {
        self + Value::new(-rhs)
    }
}

impl ops::Mul<f64> for Value {
    type Output = Value;

    fn mul(self, rhs: f64) -> Value {
        self * Value::new(rhs)
    }

}

impl ops::Div<f64> for Value {
    type Output = Value;

    fn div(self, rhs: f64) -> Value {
        self / Value::new(rhs)
    }

}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Value[{}, grad={}, op={:?}]", self.value(), self.gradient(), self.op())
    }
}






#[cfg(test)]
mod tests {
    use super::*;
   

    macro_rules! assert_approx {
        ($a:expr , $b:expr) => {
            assert!(($a - $b).abs() < 1e-8, "{} !~= {}", $a, $b);
        };
    }

    #[test]
    fn create_a_value() {
        let v = ValueData::new(10.0);
        assert_eq!(v.value, 10.0);
    }


    #[test]
    fn tanh() {
        let a = Value::from(1.0,vec![],Op::None);
        let b = a.tanh();        
        assert_eq!(b.value(), 0.7615941559557649);
        assert!(matches!(b.op(), Op::Tanh));      
    }

    #[test]
    fn exp() {
        let a = Value::from(5.0,vec![],Op::None);
        let b = a.exp();        
        assert_eq!(b.value(), 148.4131591025766);
        assert!(matches!(b.op(), Op::Exp));      
    }


    #[test]
    fn pow() {
        let a = Value::from(5.0,vec![],Op::None);
        let b = a.pow(2.0);        
        assert_eq!(b.value(), 25.0);
        assert!(matches!(b.op(), Op::Pow));      
    }

    #[test]
    fn test_arithmetic_operations_serie1() {
        // Test arithmetic operations
        let a = Value::from(2.0, vec![], Op::None);
        let b = Value::from(3.0, vec![], Op::None);

        let result_add = a.clone() + b.clone();
        let result_sub = a.clone() - b.clone();
        let result_mul = a.clone() * b.clone();
        let result_div = a.clone() / b.clone();

        // Assert the results of arithmetic operations
        assert_eq!(result_add.value(), 5.0);
        assert_eq!(result_sub.value(), -1.0);
        assert_eq!(result_mul.value(), 6.0);
        assert_eq!(result_div.value(), 2.0 / 3.0);
      
    }

    #[test]
    fn test_arithmetic_operations_serie2() {
        // Test arithmetic operations
        let a = Value::from(2.0, vec![], Op::None);
        let b = 3.0;

        let result_add = a.clone() + b;
        let result_sub = a.clone() - b;
        let result_mul = a.clone() * b;
        let result_div = a.clone() / b;

        // Assert the results of arithmetic operations
        assert_eq!(result_add.value(), 5.0);
        assert_eq!(result_sub.value(), -1.0);
        assert_eq!(result_mul.value(), 6.0);
        assert_eq!(result_div.value(), 2.0 / 3.0);
      
    }

    #[test]
    fn test_arithmetic_operations_serie3() {
        // Test arithmetic operations
        let a = 2.0;
        let b = Value::from(3.0, vec![], Op::None);

        let result_add = a + b.clone();
        let result_sub = a - b.clone();
        let result_mul = a * b.clone();
        let result_div = a / b.clone();

        // Assert the results of arithmetic operations
        assert_eq!(result_add.value(), 5.0);
        assert_eq!(result_sub.value(), -1.0);
        assert_eq!(result_mul.value(), 6.0);
        assert_eq!(result_div.value(), 2.0 / 3.0);
      
    }

    #[test]
    fn topological_order() {
        let v1 = Value::from(5.0,vec![],Op::None);
        let v2 = Value::from(1.0,vec![],Op::None);
        let v3 = v1 + v2; // 6
        //let v4 = 2.0 * v3.clone(); // 12   need to wrap values in Value()!!!!
        //let v5 = 3.0 * v3.clone(); // 18
        //let v4 = v3.clone() * 2.0; // 12
        //let v5 = v3.clone() * 3.0; // 18
        let v4 = v3.clone() * Value::from(2.0,vec![],Op::None); // 12
        let v5 = v3.clone() * Value::from(3.0,vec![],Op::None); // 18
        let v6 = v4 * v5; // 216

        //v6.backward();
        
        println!("{}",Value::export_graph(&v6));

        let order = Value::topological_sort(&v6);

        for p in order.clone(){
            println!("{}", p);
        }      
        assert_eq!(order.len(), 8);

       

    }

    #[test]
    fn abc_backpropagation1()
    {
        let a = Value::from(2.0, vec![], Op::None);
        let b = Value::from(3.0, vec![], Op::None);
        let c = Value::from(10.0, vec![], Op::None);
        let d = a.clone() * b.clone() + c.clone();
      
        d.backward();
        println!("{}",Value::export_graph(&d));
        assert_approx!(a.gradient(), 3.0);
        assert_approx!(b.gradient(), 2.0);
        assert_approx!(c.gradient(), 1.0);

    }
    
    #[test]
    fn abc_backpropagation2()
    {
        let a = Value::from(2.0, vec![], Op::None);
        let b = - Value::from(3.0, vec![], Op::None);
        let c = Value::from(10.0, vec![], Op::None);
        let d = a.clone() * b.clone() + c.clone();

        d.backward();
        println!("{}",Value::export_graph(&d));

        assert_approx!(a.gradient(), -3.0);
        assert_approx!(b.gradient(), 2.0);
        assert_approx!(c.gradient(), 1.0);

    }  

    #[test]
    fn abc_backpropagation3()
    {
        let a = Value::from(2.0, vec![], Op::None);
        let b = Value::from(- 3.0, vec![], Op::None);
        let c = Value::from(10.0, vec![], Op::None);
        let d = a.clone() * b.clone() + c.clone();

        d.backward();
        println!("{}",Value::export_graph(&d));

        assert_approx!(a.gradient(), -3.0);
        assert_approx!(b.gradient(), 2.0);
        assert_approx!(c.gradient(), 1.0);

    }

    #[test]
    fn abc_backpropagation4()
    {
        let a = Value::from(2.0, vec![], Op::None);
        let b = Value::from(3.0, vec![], Op::None);
        let c = Value::from(10.0, vec![], Op::None);
        let d = a.clone() * b.clone() - c.clone();

        d.backward();
        println!("{}",Value::export_graph(&d));

        assert_approx!(a.gradient(), 3.0);
        assert_approx!(b.gradient(), 2.0);
        assert_approx!(c.gradient(), -1.0);

    }    

    #[test]
    fn abc_backpropagation5()
    {
        let a = Value::from(2.0, vec![], Op::None);
        let b = Value::from(-3.0, vec![], Op::None);
        let c = Value::from(10.0, vec![], Op::None);
        let d = a.clone() / b.clone() + c.clone();


        d.backward();
        println!("{}",Value::export_graph(&d));
        assert_approx!(a.gradient(), -0.3333333333333333);
        assert_approx!(b.gradient(), -0.2222222222222222);
        assert_approx!(c.gradient(), 1.0);

    }

    #[test]
    fn abc_backpropagation6() {
       
        let x1 = Value::from(2.0, vec![], Op::None);
        let x2 = Value::from(0.0, vec![], Op::None);
        let w1 = Value::from(-3.0, vec![], Op::None);
        let w2 = Value::from(1.0, vec![], Op::None);
        let b = Value::from(6.88137358, vec![], Op::None);

        let x1w1 = x1 * w1.clone();
        let x2w2 = x2 * w2.clone();
        let x1w1x2w2 = x1w1 + x2w2;
        let n = x1w1x2w2 + b;
        let o = n.tanh();
        o.backward();

        println!("{}", o);

        assert_approx!(w1.gradient(), 1.0);
        assert_approx!(w2.gradient(), 0.0);
    }

}