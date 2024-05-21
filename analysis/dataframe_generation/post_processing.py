from analysis.dataframe_generation.compile_task_results import df_la, df_nj, df_performance_la

df_nj["la_slope"] = 0.0

for subject_id in df_nj.subject_id.unique():
    for plane in df_nj.plane.unique():
        la_slope = df_performance_la[(df_performance_la.subject_id == subject_id) &
                                     (df_performance_la.plane == plane)]["slope"].mean()
        q_curr_nj = (df_nj.subject_id == subject_id) & (df_nj.plane == plane)
        df_nj.loc[q_curr_nj, "la_slope"] = la_slope

df_la = df_la[df_la.stim_ele < 50]

df_nj = df_nj.dropna(subset=["stim_number", "resp_number", "la_slope"])

df_nj = df_nj[df_nj.resp_number < 7]
