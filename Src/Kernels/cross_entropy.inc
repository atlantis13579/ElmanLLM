// cross_entropy scalar implementation.
int rows=P(11);int columns=P(12);float total=0.0f;
for(int row=0;row<rows;++row){float maximum_value=-EL_INF;for(int j=0;j<columns;++j)maximum_value=EL_MAX(maximum_value,A(row*columns+j));
    float denominator=0.0f;for(int j=0;j<columns;++j)denominator+=EL_EXP(A(row*columns+j)-maximum_value);
    int target=IB(row);if(target>=0&&target<columns)total+=EL_LOG(denominator)+maximum_value-A(row*columns+target);else total+=EL_INF;
}
W(i,total/float(rows));
