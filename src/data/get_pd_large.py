def pd_result_large:
    pd_JH_data=pd.read_csv('/home/mitesh/COVID19/EDS_COVID/data/processed/COVID_relational_confirmed.csv',sep=';',parse_dates=[0])
    pd_JH_data=pd_JH_data.sort_values('date',ascending=True).copy()

        #test_structure=pd_JH_data[((pd_JH_data['country']=='US')|
        #                  (pd_JH_data['country']=='Germany'))]

    pd_result_larg=calc_filtered_data(pd_JH_data)
    pd_result_larg=calc_doubling_rate(pd_result_larg)
    pd_result_larg=calc_doubling_rate(pd_result_larg,'confirmed_filtered')


    mask=pd_result_larg['confirmed']>100
    pd_result_larg['confirmed_filtered_DR']=pd_result_larg['confirmed_filtered_DR'].where(mask, other=np.NaN)
    pd_result_larg.to_csv('/home/mitesh/COVID19/EDS_COVID/data/processed/COVID_final_set.csv',sep=';',index=False)
