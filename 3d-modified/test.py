teams_list = ["Man Utd", "Man City", "T Hotspur"]

row_format = "{:>15}" * 4
print(row_format.format("", *teams_list))
