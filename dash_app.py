from dash import Dash, dcc, html, Output, Input
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
from dash import html
from helpers import read_json

json_data = read_json(r"data/topic_visualization.json")
mapping = {
    "Topic 0: finland_in_finns": "The Role of Finland in Finnish Society",
    "Topic 1: the_economic_housing": "The State of Economic Housing",
    "Topic 2: taxation_tax_be": "Income Taxation and Its Impact",
    "Topic 3: democracy_the_media": "The Relationship Between Democracy, Media, and Parliament",
    "Topic 4: food_production_fur": "Food Production and the Use of Fur in Meat Industry",
    "Topic 5: school_students_education": "Education for School Students",
    "Topic 6: racism_immigration_immigrants": "Racism, Immigration, and the Immigrant Experience",
    "Topic 7: health_care_elderly": "Healthcare Services for the Elderly",
    "Topic 8: child_children_parents": "The Role of Children, Parents, and Families in Society",
    "Topic 9: society_culture_life": "Cultural and Social Life in Society",
    "Topic 10: salary_work_unemployment": "Salary, Employment, and Unemployment",
    "Topic 11: drug_use_cannabis": "Drug Use and the Legalization of Cannabis",
    "Topic 12: climate_change_nature": "Climate Change and Its Impact on Nature and Forests",
    "Topic 13: speech_freedom_discussion": "Freedom of Speech and Social Discussions",
    "Topic 14: gender_genders_two": "Gender, Sexuality, and the Concept of Two Genders",
    "Topic 15: religion_religious_teaching": "Religious Teaching and the Role of the Church",
    "Topic 16: yes_good_nothing": "General Feedback and Miscellaneous Thoughts",
    "Topic 17: sick_week_kela": "Sick Leave and Kela Services",
    "Topic 18: perspectives_independent_new": "Independent Perspectives and New Experiences",
    "Topic 19: phones_cell_schools": "Mobile Phones in Schools: Pros and Cons",
    "Topic 20: conscription_service_women": "Women in the Military: Conscription and Service",
    "Topic 21: corona_vaccine_media": "The Role of the Media in Reporting on the Corona Vaccine and Disease",
    "Topic 22: sitra_survey_sheltered": "Sitra's Survey on Sheltered Work and Its Relevance",
    "Topic 23: yles_news_yle": "News Coverage and Programs by YLE",
    "Topic 24: president_elected_jussi": "Election of Jussi Halla-aho as President",
    "Topic 25: future_hope_about": "Hope for the Future",
    "Topic 26: clear_concise_short": "Clear and Concise Communication",
    "Topic 27: intelligence_artificial_robotization": "Artificial Intelligence, Robotization, and Robotics",
    "Topic 28: strike_right_strikes": "The Right to Strike and the Importance of Strikes",
}
df = pd.DataFrame(json_data["points"])
df = df[~(df["topic"] == -1)]
df["id"] = df.index

df["new_topic"] = df["topic_label"].map(mapping)

df_com = pd.read_csv(r"data/global_topic_google_labeled.csv")
df_com["id"] = df_com.index

fig = px.scatter(
    df,
    x="x",
    y="y",
    color="new_topic",
    hover_data=["document", "topic_label", "new_topic"],
)
fig.update_layout(
    plot_bgcolor="rgb(0, 0, 0)",  # Dark background for the plot area
    paper_bgcolor="rgb(0, 0, 0)",  # Dark background for the entire figure
    font=dict(color="white"),  # White font color for text
    # title=dict(color='white'),  # White title color
    xaxis=dict(
        showgrid=False, color="white"
    ),  # Hide grid and set x-axis color to white
    yaxis=dict(
        showgrid=False, color="white"
    ),  # Hide grid and set y-axis color to white
)
#################################
# Accessing the maps and the issues
comms = read_json(r"data/comment_issue_map.json")
recoms = read_json(r"data/recommendation_issue_map.json")
issues = read_json(r"data/all_issues.json")
original = read_json(r"data/issues_w_id_gemini.json")

# Preparing for matching the topics
keys = list(original.keys())

keys_edited = []
for text in keys:
    result = text.split("_", 1)
    keys_edited.append(result[1])

topic_list = list(df["topic_label"].value_counts().index)

topic_list_edited = []
for i in range(len(topic_list)):
    result = topic_list[i].split(": ")
    topic = result[1]
    topic_list_edited.append(topic)

#################################
# app = Dash(__name__)

# dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
# app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc_css])
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

app.layout = html.Div(
    [
        dcc.Graph(id="scatter-plot", figure=fig),
        html.Div(id="click-output"),  # クリックしたデータを表示するためのDiv
    ],
)


@app.callback(Output("click-output", "children"), Input("scatter-plot", "clickData"))
def display_click_data(clickData):
    if clickData is None:
        return "Info on the clicked point"

    point_info = clickData["points"][0]
    # x_value = point_info['x']
    # y_value = point_info['y']
    document = point_info["customdata"][0]  # 'document' の情報が含まれている
    topic_label = point_info["customdata"][1]  # 'topic_label' の情報
    new_topic = point_info["customdata"][2]

    result = topic_label.split(": ")
    topic = result[1]

    issues_list = []
    recoms_list = []

    id = df_com[df_com["Document"] == document]["id"].values[0]
    try:
        issue_id = comms[str(id)]
        for i in issue_id:
            issues_list.append(issues[i])
            recoms_list.append(recoms[str(i)])

        return (
            html.B(f"Selected topic:"),
            html.Br(),
            f"{new_topic}",
            html.Br(),
            html.Br(),
            html.B(f"Selected comment:"),
            html.Br(),
            f"{document}",
            html.Br(),
            html.Br(),
            html.B(f"Issues identified by AI for this comment:"),
            html.Br(),
            f"{', '.join(map(str, issues_list[:2]))}",
            # f"{issues_list[:2]}",
            html.Br(),
            html.Br(),
            html.B(f"Recommendations:"),
            html.Br(),
            f"{', '.join(map(str, recoms_list[:1][0][:2]))}",
        )
    except:
        for i in range(len(keys)):
            if topic in keys_edited[i]:
                topic_issues = original[keys[i]][0]["Issues"]
                for j in range(len(topic_issues)):
                    return [
                        f"Selected topic: {new_topic}",  # id: {id}, issue_id: {issue_id}.",
                        html.Br(),
                        f"Issues identified by AI for this topic: {topic_issues[j]['Issue'][:2]}.",
                    ]
            else:
                return None  # "No applicable output."

    # return topic_label
    # return f"Clicked point - x: {x_value}, y: {y_value}, topic-label: {topic_label}, document: {document}"
    # want to return the issues!!


if __name__ == "__main__":
    app.run_server(debug=True, port=8050)

# Dash is running on http://127.0.0.1:8050/
