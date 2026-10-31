// softmax scalar implementation.
int axis=P(64);int inner=P(19+axis);int length=P(11+axis);int base=(i/(length*inner))*length*inner+i%inner;
float maximum_value=-EL_INF;for(int j=0;j<length;++j)maximum_value=EL_MAX(maximum_value,A(base+j*inner));
float denominator=0.0f;for(int j=0;j<length;++j)denominator+=EL_EXP(A(base+j*inner)-maximum_value);
W(i,EL_EXP(A(i)-maximum_value)/denominator);
