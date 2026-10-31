// dropout scalar implementation.
float value=A(i);
if(P(67)!=0){float probability=F(66);value=probability>=1.0f?0.0f:(elman_random(i,P(64),P(65))>=probability?value/(1.0f-probability):0.0f);}
W(i,value);
