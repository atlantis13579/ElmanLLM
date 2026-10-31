// index_select_grad scalar implementation.
int axis=P(64);int inner=EL_OSTRIDE(axis);int length=P(2+axis);int selected=(i/inner)%length;
int count=EL_COUNT(27,28);float value=0.0f;
for(int j=0;j<count;++j)if(IB(j)==selected)value+=A((i/(length*inner))*count*inner+j*inner+i%inner);
W(i,value);
