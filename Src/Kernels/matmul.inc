// matmul scalar implementation.
int rows=P(2+P(1)-2);int columns=P(2+P(1)-1);int row=(i/columns)%rows;int column=i%columns;int batch=i/(rows*columns);
int abase=EL_MATRIX_BATCH(batch,10,11,19);int bbase=EL_MATRIX_BATCH(batch,27,28,36);
int shared_size=P(11+P(10)-1);float value=0.0f;
for(int k=0;k<shared_size;++k)value+=A(abase+row*shared_size+k)*B(bbase+k*columns+column);
W(i,value);
