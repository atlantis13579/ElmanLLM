// index_select scalar implementation.
int axis=P(64);int inner=P(19+axis);int output_length=P(2+axis);
int selected=IB((i/inner)%output_length);
int source=(i/(inner*output_length))*P(11+axis)*inner+selected*inner+i%inner;
if(selected>=0&&selected<P(11+axis)){COPY_A(i,source);}else{W(i,0.0f);}
