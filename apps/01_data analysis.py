# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "great-tables==0.17.0",
#     "marimo",
#     "matplotlib==3.10.3",
#     "numpy==2.3.1",
#     "openpyxl==3.1.5",
#     "pandas==2.3.0",
#     "plotly==6.2.0",
#     "spacy==3.8.7",
#     "wordcloud==1.9.4",
# ]
# ///

import marimo

__generated_with = "0.14.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import pandas as pd
    from collections import defaultdict, Counter
    import numpy as np
    from great_tables import GT

    import spacy
    from spacy.lang.en.stop_words import STOP_WORDS

    from wordcloud import WordCloud

    import matplotlib.pyplot as plt
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.io as pio
    pio.templates.default = "simple_white"

    return (
        Counter,
        GT,
        STOP_WORDS,
        WordCloud,
        defaultdict,
        mo,
        np,
        pd,
        plt,
        px,
        spacy,
    )


@app.cell
def _(pd):
    data = pd.read_excel("data\\survey_data.xlsx")
    countries = pd.read_csv("data\\iso-country-codes.csv")
    clusters = pd.read_csv("data\\role_clusters.csv")
    return clusters, countries, data


@app.cell
def _():
    columns = [
        'id',
        'start_time',
        'completion_time',
        'email',
        'name',
        'language',
        'last_mod_time',
        'office',
        'country',
        'role',
        'general_understanding_rate',
        'current_past_use',
        'current_past_use_description',
        'challenges_addressed',
        'community_barriers',
        'vulnerable_impacts',
        'explo_interest',
        'explo_interest_other',
        'local_successful_uses',
        'participate_interest',
        'involvement',
        'familiar_tools',
        'sector_specific_tools',
        'existing_partners',
        'partnership_value',
        'local_initiatives',
        'local_initiatives_desc',
        'short_term_opportunities',
        'long_term_opportunities',
        'current_resources',
        'concerns',
        'final_thoughts'
    ]
    return (columns,)


@app.cell
def _(clusters, columns, countries, pd):
    # Cleaning up the dataset and preparing for quantitative analysis
    # Categorizing by countries and roles

    def clean_data(df):
        data = df.copy()
        # Cleaning up the country names, making them consistent with the ALPHA 3 codes
        data.columns = columns
        data["country"] = data.country.fillna("Ireland") # Head Office == Ireland (for mapping)
        data["country"] = data.country.replace("Republic of Sudan-Port Sudan ","Sudan (the)")
        data["country"] = data.country.replace("Sudan","Sudan (the)")
        data["country"] = data.country.replace("Ethiopia ","Ethiopia")
        data["country"] = data.country.replace("Tana River","Kenya")
        data["country"] = data.country.replace("South Korea","Korea (the Republic of)")
        data["country"] = data.country.replace("Congo, Democratic Republic of the","Congo (the Democratic Republic of the)")
        data["country"] = data.country.replace("Syria","Syrian Arab Republic")
        data["country"] = data.country.replace("Central African Republic","Central African Republic (the)")
        data["country"] = data.country.replace("Niger","Niger (the)")

        # Add ISO-3 column
        data_country_codes = pd.merge(
            left=data,
            right=countries,
            left_on="country",
            right_on="en"
        ).drop(["en", "numeric"], axis=1)

        # Role clusterization
        data_roles = pd.merge(
            left=data_country_codes,
            right=clusters,
            on="id"
        ).drop(["role_cluster", "role_y"], axis=1)
        data_roles.rename(columns={"role_x":"role"}, inplace=True)

        return data_roles

    return (clean_data,)


@app.cell
def _(clean_data, data):
    df = clean_data(data)
    # df.to_csv("cleaned_data.csv", index=False)
    return (df,)


@app.cell
def _(mo):
    mo.vstack(
        [
            mo.md("# **Digital Technologies and AI Survey**"),
            mo.md("## Concern Worldwide | 2025"),
            mo.md("<br>")
        ],
        align="center"
    )
    return


@app.cell
def _(df, mo):
    mo.vstack([
        mo.md("## **General Overview**"),
        mo.md(f"### Total Number of Responses: **{len(df)}**"),
        mo.md(f"#### Of which **{len(df[df.office != "Headquarters"])}** from **Country Offices** and **{len(df[df.office == "Headquarters"])}** from **Head Office**")
    ])
    return


@app.cell
def _(px):
    colors=px.colors.qualitative.Pastel
    return (colors,)


@app.cell
def _(colors, df, mo, px):
    pie_1 = px.pie(
        df,
        names="office",
        color_discrete_sequence=colors
    )
    pie_1.update_traces(hoverinfo="label+percent", textinfo="label+value+percent", showlegend=False)

    pie_2 = px.pie(
        df,
        names="language",
        color_discrete_sequence=colors
    )
    pie_2.update_traces(hoverinfo="label+percent", textinfo="label+value+percent", showlegend=False)

    pie_1 = mo.ui.plotly(pie_1)
    pie_2 = mo.ui.plotly(pie_2)
    return pie_1, pie_2


@app.cell
def _(mo, pie_1, pie_2):
    mo.hstack([pie_1, pie_2])
    return


@app.cell
def _(mo):
    barplot_1_dropdown = mo.ui.dropdown(
        options={"Country":"country", "Sector":"role_category"},
        value="Country",
        label="Show by"
    )
    return (barplot_1_dropdown,)


@app.cell
def _(barplot_1_dropdown, colors, df, mo, px):
    grouped_df = df.groupby([f"{barplot_1_dropdown.value}"]).count().reset_index()

    barplot_1 = px.bar(
        grouped_df.sort_values(by="id"),
        y=f"{barplot_1_dropdown.value}",
        x="id",
        labels={f"{barplot_1_dropdown.value}":"","id":"Number of responses"},
        color_discrete_sequence=colors
    )
    barplot_1_title = mo.md(f"## Total Number of Responses by {barplot_1_dropdown.selected_key}")
    barplot_1 = mo.ui.plotly(barplot_1)
    return barplot_1, barplot_1_title


@app.cell
def _(barplot_1, barplot_1_dropdown, barplot_1_title, mo):
    mo.vstack([barplot_1_dropdown, barplot_1_title, barplot_1])
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md("## **Use of AI/Digital Tools**"),
    ])
    return


@app.cell
def _(colors, df, mo, px):
    pie_3 = px.pie(
        df,
        names="current_past_use",
        color="current_past_use",
        color_discrete_sequence=colors
    )

    pie_3.update_traces(hoverinfo="label+percent", textinfo="value+percent")

    pie_3_title = mo.md("## Respondents Currently Using AI/Digital Tools")
    pie_3=mo.ui.plotly(pie_3)
    return pie_3, pie_3_title


@app.cell
def _(colors, df, mo, pd, px):
    # Creating a temporary dataframe grouped by country with the total "Yes" and the total number of responses
    tmp_df = pd.merge(
        left=df.query("current_past_use == 'Yes'").groupby(["country"]).count().reset_index(),
        right=df.groupby(["country"])["current_past_use"].count().reset_index(),
        on="country"
    ).rename(columns={"current_past_use_y":"total_repsonses", "current_past_use_x":"current_past_use_yes"})

    tmp_df["pcg_yes"] = round((tmp_df.current_past_use_yes / tmp_df.total_repsonses) * 100,2)

    tmp_df = tmp_df[["country","current_past_use_yes","total_repsonses","pcg_yes"]].sort_values(by=["current_past_use_yes","pcg_yes"], ascending=False)

    tmp_df_ = tmp_df.copy()
    tmp_df_.columns = ["Country","Total Yes","Total Responses","% Yes"]

    table_1 = mo.ui.table(tmp_df_)

    barplot_2 = px.bar(
        tmp_df.head(10),
        y="current_past_use_yes",
        x="country",
        labels={"country":"","current_past_use_yes":"Number of people using AI/digital tools"},
        color_discrete_sequence=colors,
        # text_auto=True,
        # text="pcg_yes",
    )
    barplot_2_title = mo.md("## Top 10 Countries according to the number of people using AI/digital tools")
    barplot_2 = mo.ui.plotly(barplot_2)
    return barplot_2, barplot_2_title


@app.cell
def _(STOP_WORDS, WordCloud, df, mo, plt, spacy):
    texts = df.current_past_use_description.dropna().astype(str)
    nlp = spacy.load("en_core_web_sm")

    STOP_WORDS.update({"use", "concern", "ai", "etc"})

    def preprocess(text):
        doc = nlp(text.lower())
        return " ".join([token.lemma_ for token in doc if token.is_alpha and token.text not in STOP_WORDS])

    def replace_all(text, rep):
        for i, j in rep.items():
            text = text.replace(i, j)
        return text

    rep_dict = {"chat gpt":"chatgpt", "co pilot": "copilot", "datum":"data", "base":"basic", "creation":"create", "development":"develop", "drafting":"draft", "editing":"edit", "generation":"generate", "google workspace":"google", "improvement": "improve", "know":"knowledge", "learning":"learn", "ms":"microsoft", "personal life":"personal", "summarise":"summarize", "summary":"summarize", "writing":"write", "translation":"translate", "translation service":"translate", "translator":"translate"}

    cleaned = texts.apply(preprocess)
    cleaned = cleaned.apply(replace_all, args=(rep_dict,))

    txt = " ".join(cleaned)
    wc = WordCloud(width=600, height=300, background_color="white").generate(txt)
    plt.imshow(wc,interpolation="bilinear")
    plt.axis("off")
    # plt.show()

    wordcloud_plot = mo.as_html(plt.gca())
    wordcloud_txt = mo.md("## More frequently used terms ##")
    return preprocess, wordcloud_plot, wordcloud_txt


@app.cell
def _(
    barplot_2,
    barplot_2_title,
    mo,
    pie_3,
    pie_3_title,
    wordcloud_plot,
    wordcloud_txt,
):
    # mo.vstack([pie_3_title, pie_3, wordcloud_plot, table_1, barplot_2_title, barplot_2])
    mo.vstack([pie_3_title, pie_3, wordcloud_txt, wordcloud_plot, barplot_2_title, barplot_2])
    return


@app.cell
def _(df, mo):
    mo.vstack([
        mo.md("## **Self Assessed Proficiency with AI/Digital Tools**"),
        mo.md(f"### Average Rating: **{df.general_understanding_rate.mean():.1f}** (on a scale 1-5)")
    ])
    return


@app.cell
def _(mo):
    barplot_3_dropdown = mo.ui.dropdown(
        options={"Country":"country", "Sector":"role_category"},
        value="Country",
        label="Show by"
    )

    switch_1 = mo.ui.switch(label="Box Plot")
    return barplot_3_dropdown, switch_1


@app.cell
def _(barplot_3_dropdown, colors, df, mo, px, switch_1):
    if not switch_1.value:

        barplot_3 = px.bar(
            df.groupby(f"{barplot_3_dropdown.value}").agg({"general_understanding_rate":"mean","id":"count"}).reset_index().sort_values(by="general_understanding_rate", ascending=False),
            x=f"{barplot_3_dropdown.value}",
            y="general_understanding_rate",
            hover_data="id",
            color_discrete_sequence=colors,
            labels={"general_understanding_rate":"Average Rate", f"{barplot_3_dropdown.value}":"", "id":"Number of Responses"}
        )

        barplot_3_title = mo.md(f"## Average Rating by {barplot_3_dropdown.selected_key}")

    else:
        barplot_3 = px.box(
            df,
            x=f"{barplot_3_dropdown.value}",
            y="general_understanding_rate",
            color_discrete_sequence=colors,
            labels={"general_understanding_rate":"Self Assessed Proficiency Rate", f"{barplot_3_dropdown.value}":""}
        )

        barplot_3_title = mo.md(f"## Rating Distribution by {barplot_3_dropdown.selected_key}")

    # Alternative Method 2: Using add_shape (if add_hline doesn't work)
    barplot_3.add_shape(
        type="line",
        x0=0, x1=1, xref="paper",  # x0=0, x1=1, xref="paper" means full width
        y0=df.general_understanding_rate.mean(), y1=df.general_understanding_rate.mean(),
        line=dict(color="red", width=2, dash="dash")
    )
    # Add annotation for the line
    barplot_3.add_annotation(
        x=0.95, xref="paper",  # Position at 95% of plot width
        y=df.general_understanding_rate.mean(),
        text=f"Overall Mean: {df.general_understanding_rate.mean():.1f}",
        showarrow=False,
        bgcolor="white",
        # bordercolor="red",
        # borderwidth=1,
        font=dict(color="red", size=10)
    )
    barplot_3 = mo.ui.plotly(barplot_3)
    return barplot_3, barplot_3_title


@app.cell
def _(barplot_3, barplot_3_dropdown, barplot_3_title, mo, switch_1):
    mo.vstack([mo.hstack([barplot_3_dropdown, switch_1],widths=[1,1]), barplot_3_title, barplot_3])
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md("## **AI/Digital Solutions Implementation Priorities**")
    ])
    return


@app.cell
def _(Counter, defaultdict, df, np, pd):
    # Functions to handle ranking questions
    # Clean and parse the responses
    def parse_responses(response):
        """Parse a semicolon-separated response into individual responses"""
        if pd.isna(response) or response in ['N/A', 'n/a', 'none', 'Nothing at the moment']:
            return []

        # Split by semicolon and clean each response
        responses = []
        for res in str(response).split(';'):
            res = res.strip()
            if res and res not in ['N/A', 'n/a', 'none', '']:
                responses.append(res)
        return responses

    def create_ranking(series):
        # Parse all responses
        all_responses = []
        position_weights = defaultdict(list)
        res_counts = Counter()

        for idx, response in enumerate(series):
            responses = parse_responses(response)

            for position, res in enumerate(responses):
                all_responses.append(res)
                # Weight based on position (first mentioned = higher weight)
                position_weight = 1 / (position + 1)  # 1st = 1.0, 2nd = 0.5, 3rd = 0.33, etc.
                position_weights[res].append(position_weight)
                res_counts[res] += 1

        # Calculate rankings
        ranking_data = []

        for res, weights in position_weights.items():
            total_mentions = len(weights)
            avg_position_weight = np.mean(weights)
            total_weighted_score = sum(weights)

            # Calculate average position (1-indexed)
            positions = [i + 1 for i in range(len(weights))]
            avg_position = np.mean([1/(w) for w in weights])  # Reverse the weight calculation

            ranking_data.append({
                'Response': res,
                'Total_Mentions': total_mentions,
                'Avg_Position_Weight': avg_position_weight,
                'Total_Weighted_Score': total_weighted_score,
                'Avg_Position': avg_position,
                'Percentage_of_Responses': (total_mentions / len(df)) * 100
            })

        return ranking_data
    return (create_ranking,)


@app.cell
def _(colors, create_ranking, df, mo, pd, px):
    interest_ranking = pd.DataFrame(create_ranking(df.explo_interest)).iloc[:7,:]

    barplot_5 = px.bar(
        interest_ranking.sort_values(by="Total_Weighted_Score", ascending=True),
        y="Response",
        x="Total_Weighted_Score",
        color_discrete_sequence=colors,
        labels={"Response":"", "Total_Weighted_Score": "Totale Weighted Score"}
    )
    barplot_5.update_layout(xaxis=dict(tickformat=',d'))
    barplot_5_title = mo.md("## Ranked by total weighted score")

    barplot_5 = mo.ui.plotly(barplot_5)
    return barplot_5, barplot_5_title


@app.cell
def _(WordCloud, df, mo, np, plt, preprocess):
    explo_interest_other = df.explo_interest_other.dropna().astype(str)

    explo_interest_other_cleaned = explo_interest_other.apply(preprocess).replace("na", np.nan)
    explo_interest_other_cleaned = explo_interest_other_cleaned.dropna()
    # cleaned = cleaned.apply(replace_all)

    txt_interest = " ".join(explo_interest_other_cleaned)
    wc_interest = WordCloud(width=800, height=400, background_color="white").generate(txt_interest)
    plt.imshow(wc_interest,interpolation="bilinear")
    plt.axis("off")
    # plt.show()

    wc_interest_plot = mo.as_html(plt.gca())
    wc_interest_title = mo.md("## More frequently used terms ##")
    wc_interest_txt = mo.md("Most mentioned priorities include: data analysis, health information system, infrastructure monitoring, and report/concept notes development.")
    return wc_interest_plot, wc_interest_title, wc_interest_txt


@app.cell
def _(
    barplot_5,
    barplot_5_title,
    mo,
    wc_interest_plot,
    wc_interest_title,
    wc_interest_txt,
):
    # mo.vstack([barplot_5_title, barplot_5])
    mo.hstack([mo.vstack([barplot_5_title, barplot_5]), mo.vstack([wc_interest_title, wc_interest_plot, wc_interest_txt])], widths=[3,1])
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md("## **Main technological barriers faced by the project participants**"),
    ])
    return


@app.cell
def _(colors, create_ranking, df, mo, pd, px):
    barriers_ranking = pd.DataFrame(create_ranking(df.community_barriers)).iloc[:5,:]

    barplot_4 = px.bar(
        barriers_ranking.sort_values(by="Total_Mentions"),
        y="Response",
        x="Total_Mentions",
        color_discrete_sequence=colors,
        labels={"Response":"", "Total_Mentions": "Total Mentions"}
    )
    barplot_4_title = mo.md("## Ranked by total number of mentions")

    barplot_4 = mo.ui.plotly(barplot_4)
    return (barplot_4,)


@app.cell
def _(barplot_4, mo):
    mo.hstack([barplot_4])
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md("## **Known Local Initiatives**"),
    ])
    return


@app.cell
def _(pd):
    known_initiatives = pd.read_csv("data\\known_initiatives.csv")
    # known_initiatives
    return (known_initiatives,)


@app.cell
def _(mo):
    barplot_6_dropdown = mo.ui.dropdown(
        options={"Country":"country", "Category":"category"},
        value="Category",
        label="Show by"
    )
    return (barplot_6_dropdown,)


@app.cell
def _(barplot_6_dropdown, colors, known_initiatives, mo, px):
    barplot_6 = px.bar(
        known_initiatives.groupby([barplot_6_dropdown.value]).count().sort_values(by="id"),
        x="id",
        color_discrete_sequence=colors,
        labels={f"{barplot_6_dropdown.value}":"", "id": "Total Mentions"}
    )

    barplot_6_title = mo.md(f"## Number of known local initiatives by {barplot_6_dropdown.value}")

    barplot_6 = mo.ui.plotly(barplot_6)
    return barplot_6, barplot_6_title


@app.cell
def _(barplot_6, barplot_6_dropdown, barplot_6_title, mo):
    mo.vstack([barplot_6_dropdown, barplot_6_title, barplot_6])
    return


@app.cell
def _(known_initiatives, mo, np):
    table_2_dropdown_1 = mo.ui.multiselect(
        options=np.sort(known_initiatives.country.unique()),
        value=known_initiatives.country.unique()[:3],
        label="Filter by country",
    )

    table_2_dropdown_2 = mo.ui.multiselect(
        options=np.sort(known_initiatives.category.unique()),
        value=known_initiatives.category.unique(),
        label="Filter by category",
    )
    return table_2_dropdown_1, table_2_dropdown_2


@app.cell
def _(GT, known_initiatives, table_2_dropdown_1, table_2_dropdown_2):
    table_2 = (
        GT(known_initiatives.query(f"country == {table_2_dropdown_1.value} & category == {table_2_dropdown_2.value}").sort_values(by="country"))
        .cols_hide(["id"])
        .cols_label({"local_successful_uses":"Description", "category":"Category", "country":"Country"})
        .cols_width(cases={"country":"100px","local_successful_uses":"500px", "category":"200px"})
    )
    return (table_2,)


@app.cell
def _(mo, table_2, table_2_dropdown_1, table_2_dropdown_2):
    mo.vstack([mo.hstack([table_2_dropdown_2, table_2_dropdown_1],widths=[1,2]),table_2])
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md("## **Gauging respondents interest in participating in pilot projects or further discussions**"),
    ])
    return


@app.cell
def _(colors, df, mo, px):
    pie_4 = px.pie(
        df,
        names="participate_interest",
        color="participate_interest",
        color_discrete_sequence=colors
    )

    pie_4.update_traces(hoverinfo="label+percent", textinfo="value+percent")

    pie_4_title = mo.md("## Gauging respondents interest")
    pie_4=mo.ui.plotly(pie_4)
    return (pie_4,)


@app.cell
def _(colors, create_ranking, df, mo, pd, px):
    involvement_ranking = pd.DataFrame(create_ranking(df.involvement)).iloc[:4,:]

    barplot_7 = px.bar(
        involvement_ranking.sort_values(by="Total_Mentions"),
        y="Response",
        x="Total_Mentions",
        color_discrete_sequence=colors,
        labels={"Response":"", "Total_Mentions": "Total Mentions"}
    )

    barplot_7_title = mo.md("## Ranked by role")

    barplot_7 = mo.ui.plotly(barplot_7)
    return (barplot_7,)


@app.cell
def _(barplot_7, mo, pie_4):
    # mo.vstack([barplot_7_title,barplot_7])
    mo.hstack([barplot_7,pie_4])
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md("## **Partnerships and Resources**"),
    ])
    return


@app.cell
def _(colors, create_ranking, df, mo, pd, px):
    partnership_ranking = pd.DataFrame(create_ranking(df.partnership_value))

    barplot_8 = px.bar(
        partnership_ranking.sort_values(by="Total_Weighted_Score"),
        y="Response",
        x="Total_Weighted_Score",
        color_discrete_sequence=colors,
        labels={"Response":"", "Total_Weighted_Score": "Total Weighted Score"}
    )

    barplot_8_title = mo.md("### Most valuable partnerships")
    return barplot_8, barplot_8_title


@app.cell
def _(colors, create_ranking, df, mo, pd, px):
    resources_ranking = pd.DataFrame(create_ranking(df.current_resources)).iloc[:4,:]

    barplot_9 = px.bar(
        resources_ranking.sort_values(by="Total_Mentions"),
        y="Response",
        x="Total_Mentions",
        color_discrete_sequence=colors,
        labels={"Response":"", "Total_Mentions": "Total Mentions"}
    )

    barplot_9_title = mo.md("### Current available resources")
    return barplot_9, barplot_9_title


@app.cell
def _(barplot_8, barplot_8_title, barplot_9, barplot_9_title, mo):
    mo.hstack([mo.vstack([barplot_8_title,barplot_8]), mo.vstack([barplot_9_title,barplot_9])])
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md("## **Moving Forward**"),
    ])
    return


@app.cell
def _(colors, create_ranking, df, mo, pd, px):
    short_ranking = pd.DataFrame(create_ranking(df.short_term_opportunities)).iloc[:5,:]

    barplot_10 = px.bar(
        short_ranking.sort_values(by="Total_Mentions"),
        y="Response",
        x="Total_Mentions",
        color_discrete_sequence=colors,
        labels={"Response":"", "Total_Mentions": "Total Mentions"}
    )

    import textwrap

    # Function to wrap text
    def wrap_text(text, width=20):  # Adjust width as needed
        return '<br>'.join(textwrap.wrap(text, width=width))

    # Apply text wrapping to your DataFrame
    short_ranking['Response_wrapped'] = short_ranking['Response'].apply(wrap_text)

    # Update y-axis to handle long labels
    barplot_10.update_layout(
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(short_ranking))),
            ticktext=[wrap_text(text, 25) for text in short_ranking.sort_values(by="Total_Mentions")['Response']],
            tickfont=dict(size=10)  # Adjust font size if needed
        ),
        margin=dict(l=200)  # Increase left margin to accommodate longer labels
    )

    barplot_10_title = mo.md("### Short-terms opportunities (6 months)")
    return barplot_10, barplot_10_title, wrap_text


@app.cell
def _(colors, create_ranking, df, mo, pd, px, wrap_text):
    long_ranking = pd.DataFrame(create_ranking(df.long_term_opportunities))

    barplot_11 = px.bar(
        long_ranking.sort_values(by="Total_Weighted_Score"),
        y="Response",
        x="Total_Weighted_Score",
        color_discrete_sequence=colors,
        labels={"Response":"", "Total_Weighted_Score": "Total Weighted Score"}
    )

    # Update y-axis to handle long labels
    barplot_11.update_layout(
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(long_ranking))),
            ticktext=[wrap_text(text, 25) for text in long_ranking.sort_values(by="Total_Weighted_Score")['Response']],
            tickfont=dict(size=10)  # Adjust font size if needed
        ),
        margin=dict(l=200)  # Increase left margin to accommodate longer labels
    )

    barplot_11_title = mo.md("### Medium/Long-terms opportunities (6-24 months)")
    return barplot_11, barplot_11_title


@app.cell
def _(barplot_10, barplot_10_title, barplot_11, barplot_11_title, mo):
    mo.hstack([mo.vstack([barplot_10_title,barplot_10]), mo.vstack([barplot_11_title,barplot_11])])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
