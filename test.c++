template <class T>
struct Funcd
{
    Doub EPS;
    T &func;
    Doub f;
    Funcd(T &funcc) : EPS(1.0e-8), func(funcc) {}
    Doub operator()(VecDoub_I &x)
    {
        return f = func(x);
    }
    void df(VecDoub_I &x, VecDoub_O &df)
    {
        Int n = x.size();
        VecDoub xh = x;
        Doub fold = f;
        for (Int j = 0; j < n; j++)
        {
            Doub temp = x[j];
            Doub h = EPS * abs(temp);
            if (h == 0.0)
                h = EPS;
            xh[j] = temp + h;
            h = xh[j] - temp;
            Doub fh = operator()(xh);
            xh[j] = temp;
            df[j] = (fh - fold) / h;
        }
    }
};