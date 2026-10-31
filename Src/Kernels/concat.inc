// concat scalar implementation.
int axis=P(64);int inner=P(19+axis);int full=P(2+axis);int coordinate=(i/inner)%full;int outer=i/(inner*full);int rest=i%inner;
if(coordinate<P(11+axis)){int source=outer*P(11+axis)*inner+coordinate*inner+rest;COPY_A(i,source);}
else{int source=outer*P(28+axis)*inner+(coordinate-P(11+axis))*inner+rest;if(P(253)==0)W(i,B(source));else WI(i,IB(source));}
