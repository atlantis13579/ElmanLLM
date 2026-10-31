// reduce_grad scalar implementation.
int axis=P(64);
int inner=axis<0?1:P(19+axis);
int length=axis<0?P(0):P(11+axis);
int target=axis<0?0:(i/(length*inner))*inner+i%inner;
float value=C(target);
int kind=P(66);
if(kind==1)value/=float(length);
if(kind==2||kind==3){
    float extremum=B(target);
    int base=axis<0?0:(target/inner)*length*inner+target%inner;
    int matches=0;
    for(int j=0;j<length;++j)if(A(base+j*inner)==extremum)++matches;
    value=A(i)==extremum?value/float(matches):0.0f;
}
if(kind==4){
    int base=axis<0?0:(target/inner)*length*inner+target%inner;
    float product=1.0f;
    for(int j=0;j<length;++j)if(base+j*inner!=i)product*=A(base+j*inner);
    value*=product;
}
W(i,value);
