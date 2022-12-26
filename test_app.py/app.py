from shiny import *
from api_utils import request_prediction
from constant import URL


app_ui = ui.page_fluid(
    ui.panel_title(title='Product recommendation predictor'),
    
    ui.layout_sidebar(
    ui.panel_sidebar(
        ui.navset_tab(
        # left hand side ----
                            ui.nav_menu("Project items", 
                                        ui.nav("Project Description", #"LET talk about Project Description",
                                               value="project_desc_sidebar"),
                                        ui.nav("Modelling Data", #"Describe the Modelling Data",
                                            value='model_data_sidebar'),
                                        ui.nav("Prediction", value="prediction_sidebar"),
                                    ),
                            id="sidebar_id",
    
                            ),
        ),
    
    ui.panel_main(ui.output_ui(id="main_output_show")                  
                  ),
    )
)


def server(input, output, session):
    @output
    @render.ui
    def main_output_show():
        selected_sidebar_dropdown = input.sidebar_id()
        if selected_sidebar_dropdown == "prediction_sidebar":
            
            
            
            return [
                    ui.row(ui.column(8,
                             ui.input_text_area(id="review", label=f"Write a product review", 
                                                width="100%", height="200px", autocomplete="on",
                                                placeholder="Write product review here"
                                                ),
                             ui.tags.br(),
                             ui.input_action_button(id="predict", label="Prediction")
                             ), 
                             ui.column(4,
                                       ui.h4("Review Prediction"), 
                                       ui.output_text(id='prediction_desc'))
                   )
                ]
        elif selected_sidebar_dropdown == "project_desc_sidebar":
            return "This is a NLP project for predicting whether a product will be recommended based on review text"
        else:
            return "Describe model data"
    
    @output
    @render.text    
    @reactive.event(input.predict, ignore_none=True)    
    def prediction_desc():
        review = input.review()
        prediction = request_prediction(URL=URL, review_data=review)
        return prediction


app = App(app_ui, server)

