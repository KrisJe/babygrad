
mod engine;
mod nn;

use engine::Value;
use nn::Neuron;


fn main() {   
    println!("Hello, this is babygrad!");
    println!("Run: cargo test -- --nocapture");
    println!("Go to: https://dreampuf.github.io/GraphvizOnline");
    println!("Copy paste the Graphviz dot output to visualise.");
    /*
    strict graph {
    rankdir=RL;
    node [shape=record,colorscheme=set28];
    84 [label="{* | 216.00 | 0.00}", color=2];
    81 [label="{* | 12.00 | 0.00}", color=2];
    79 [label="{+ | 6.00 | 0.00}", color=1];
    77 [label="{v1 | 5.00 | 0.00}", color=0];
    79 -- 77;
    78 [label="{v2 | 1.00 | 0.00}", color=0];
    79 -- 78;
    81 -- 79;
    80 [label="{vx | 2.00 | 0.00}", color=0];
    81 -- 80;
    84 -- 81;
    83 [label="{* | 18.00 | 0.00}", color=2];
    79 [label="{+ | 6.00 | 0.00}", color=1];
    77 [label="{v1 | 5.00 | 0.00}", color=0];
    79 -- 77;
    78 [label="{v2 | 1.00 | 0.00}", color=0];
    79 -- 78;
    83 -- 79;
    82 [label="{vy | 3.00 | 0.00}", color=0];
    83 -- 82;
    84 -- 83;
    } 
    */
}

