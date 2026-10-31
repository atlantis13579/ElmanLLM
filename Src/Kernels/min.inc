// min scalar implementation.
int axis=P(64);
int inner=axis<0?1:P(19+axis);
int length=axis<0?EL_COUNT(10,11):P(11+axis);
int base=axis<0?0:(i/inner)*length*inner+i%inner;
float value=EL_INF;
for(int j=0;j<length;++j){float x=A(base+j*inner);value=EL_MIN(value,x);}
W(i,value);
