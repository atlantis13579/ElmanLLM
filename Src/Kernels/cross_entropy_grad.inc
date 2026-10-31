// cross_entropy_grad scalar implementation.
int rows=P(11);int columns=P(12);int row=i/columns;int column=i%columns;float maximum_value=-EL_INF;
for(int j=0;j<columns;++j)maximum_value=EL_MAX(maximum_value,A(row*columns+j));
float denominator=0.0f;for(int j=0;j<columns;++j)denominator+=EL_EXP(A(row*columns+j)-maximum_value);
float value=EL_EXP(A(i)-maximum_value)/denominator-float(column==IB(row));W(i,value*C(0)/float(rows));
