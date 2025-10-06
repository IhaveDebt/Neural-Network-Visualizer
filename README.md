/**
 * neural_visualizer.ts
 *
 * A simple neural network visualizer that trains a small feedforward NN and logs weight updates.
 * This prototype helps visualize how weights evolve over time.
 *
 * Run:
 *   ts-node src/neural_visualizer.ts
 */

class NeuralNet {
  input:number; hidden:number; output:number;
  w1:number[][]; w2:number[][];
  lr:number=0.1;

  constructor(input:number, hidden:number, output:number){
    this.input=input; this.hidden=hidden; this.output=output;
    this.w1=Array.from({length:input},()=>Array.from({length:hidden},()=>Math.random()-0.5));
    this.w2=Array.from({length:hidden},()=>Array.from({length:output},()=>Math.random()-0.5));
  }

  sigmoid(x:number){ return 1/(1+Math.exp(-x)); }
  dsigmoid(x:number){ return x*(1-x); }

  forward(input:number[]): {h:number[], o:number[]} {
    const h = this.w1[0].map((_,j)=>this.sigmoid(input.reduce((s,v,i)=>s+v*this.w1[i][j],0)));
    const o = this.w2[0].map((_,k)=>this.sigmoid(h.reduce((s,v,j)=>s+v*this.w2[j][k],0)));
    return {h,o};
  }

  train(data:{x:number[],y:number[]}[], epochs:number){
    for(let e=0;e<epochs;e++){
      for(const {x,y} of data){
        const {h,o}=this.forward(x);
        const oErr=y.map((t,k)=>t-o[k]);
        const oDelta=oErr.map((err,k)=>err*this.dsigmoid(o[k]));
        const hErr=this.w2.map((col,j)=>col.reduce((s,w,k)=>s+w*oDelta[k],0));
        const hDelta=hErr.map((err,j)=>err*this.dsigmoid(h[j]));
        for(let j=0;j<this.hidden;j++){
          for(let k=0;k<this.output;k++){
            this.w2[j][k]+=h[j]*oDelta[k]*this.lr;
          }
        }
        for(let i=0;i<this.input;i++){
          for(let j=0;j<this.hidden;j++){
            this.w1[i][j]+=x[i]*hDelta[j]*this.lr;
          }
        }
      }
      if(e%100===0) console.log(`Epoch ${e}: Sample output ${this.forward(data[0].x).o}`);
    }
  }
}

// Demo: XOR problem
const nn=new NeuralNet(2,4,1);
const data=[
  {x:[0,0],y:[0]},
  {x:[0,1],y:[1]},
  {x:[1,0],y:[1]},
  {x:[1,1],y:[0]},
];
nn.train(data,1000);
console.log('Trained results:');
for(const d of data) console.log(d.x, '→', nn.forward(d.x).o.map(v=>v.toFixed(2)));
