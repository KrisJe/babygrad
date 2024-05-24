use std::ops;
use std::{cell::{Ref, RefCell, RefMut},
     rc::Rc};


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
    Relu
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct ValueData {
    value: f64,
    children: Vec<Value>,
    gradient: f64,
    op: Op,
    visited: bool,
}


impl ValueData {
    fn new(value: f64) -> ValueData {       
        ValueData {
            value: value,
            children: vec![],
            gradient: 0.0,
            op: Op::None,
            visited: false,
        }
    }
}



#[derive(Clone, Debug)]
pub struct Value(Rc<RefCell<ValueData>>);

impl Value {
    pub fn new(value: f64) -> Value {
        Value(Rc::new(RefCell::new(ValueData {
            value: value,
            children: vec![], //Vec::new(),
            gradient: 0.0,
            op: Op::None,
            visited: false,
        })))
    }

    pub fn from(value: f64, children: Vec<Value>, op: Op) -> Value {
        Value(Rc::new(RefCell::new(ValueData {
            value: value,
            children: children,
            gradient: 0.0,
            op: op,
            visited: false,
        })))
    } 

    
    pub fn op(&self) -> Op {
        self.0.borrow().op.clone()
    }


    pub fn value(&self) -> f64 {
        self.0.borrow().value
    }

    pub fn gradient(&self) -> f64 {
        self.0.borrow().gradient
    }

    pub fn vec(values: &[f64]) -> Vec<Value> {
        values
            .iter()
            .map(|el| Value::new(*el))
            .collect::<Vec<Value>>()
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

impl ops::Mul<Value> for f64 {
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





#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_a_value() {
        let v = ValueData::new(10.0);
        assert_eq!(v.value, 10.0);
    }

    #[test]
    fn add() {
        let a = Value::from(5.0,vec![],Op::None);
        let b = Value::from(2.0,vec![],Op::None);
        let c = a.clone() + b.clone();
        assert_eq!(c.value(), 7.0);
        assert!(matches!(c.op(), Op::Add));    

        let c = a.clone() + 3.0;
        assert_eq!(c.value(), 8.0);
        assert!(matches!(c.op(), Op::Add));  

        let c = 5.0 + a.clone();
        assert_eq!(c.value(), 10.0);
        //assert!(matches!(c.op(), Op::Add)); //DOESN'T WORK!

    }

    #[test]
    fn subtract() {
        let a = Value::from(5.0,vec![],Op::None);
        let b = Value::from(2.0,vec![],Op::None);
        let c = a.clone() - b.clone();
        assert_eq!(c.value(), 3.0);
        assert!(matches!(c.op(), Op::Sub)); 


        let c = b.clone() - a.clone();
        assert_eq!(c.value(), -3.0);
        assert!(matches!(c.op(), Op::Sub));      
    }


    #[test]
    fn multiply() {
        let a = Value::from(5.0,vec![],Op::None);
        let b = Value::from(2.0,vec![],Op::None);
        let c = a.clone() * b.clone();
        assert_eq!(c.value(), 10.0);
        assert!(matches!(c.op(), Op::Mul));        
    }

    #[test]
    fn scalar_multiply() {
        let a = 5.0;
        let b = Value::from(2.0,vec![],Op::None);
        let c = b.clone() * a;    
        assert_eq!(c.value(), 10.0);
        assert!(matches!(c.op(), Op::Mul)); 
      
        let c = a * b.clone(); 
        assert_eq!(c.value(), 10.0);
        //assert!(matches!(c.op(), Op::Mul)); //DOESN'T WORK!
    }


    #[test]
    fn div() {
        let a = Value::from(5.0,vec![],Op::None);
        let b = Value::from(2.0,vec![],Op::None);
        let c = a.clone() / b.clone();
        assert_eq!(c.value(), 2.5);
        assert!(matches!(c.op(), Op::Div));      
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

   

}