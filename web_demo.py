import gradio as gr
from utils import *

gr.Chatbot.postprocess = postprocess

with gr.Blocks(title="ChatGLM ArtAgent") as demo:
    gr.HTML("""<h1 align="center">ChatGLM ArtAgent</h1>""")
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot().style(height=640)
            with gr.Box():
                with gr.Row():
                    with gr.Column(scale=8):
                        user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=3).style(
                            container=False)
                    with gr.Column(scale=1):
                        submitBtn = gr.Button("Chat", variant="primary")
                        drawBtn = gr.Button("Generate Image")
        with gr.Column(scale=3):
            with gr.Group():
                result_gallery = gr.Gallery(label='Output', show_label=False).style(preview=True)
            with gr.Row():
                with gr.Accordion(label="Stable Diffusion"):
                    with gr.Column():
                        # clearBtn = gr.Button("Clear Gallery")
                        sd_width = gr.Slider(480, 768, value=512, step=32, label="Width", interactive=True)
                        sd_height = gr.Slider(480, 768, value=512, step=32, label="Height", interactive=True)
                        sd_steps = gr.Slider(8, 40, value=32, step=4, label="Steps", interactive=True)
                with gr.Accordion(label="ChatGLM-6B"):
                    with gr.Column():
                        # emptyBtn = gr.Button("Clear History")
                        max_length = gr.Slider(0, 4096, value=2048, step=64.0, label="Maximum length", interactive=True)
                        top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
                        temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

    history = gr.State([])
    result_list = gr.State([])

    submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history], [chatbot, history],
                    show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    drawBtn.click(sd_predict, [user_input, chatbot, max_length, top_p, temperature, history, sd_width, sd_height, sd_steps, result_list],
                     [chatbot, history, result_list, result_gallery], show_progress=True)
    drawBtn.click(reset_user_input, [], [user_input])

    # emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)
    # clearBtn.click(clear_gallery, outputs=[result_list, result_gallery], show_progress=True)

demo.queue().launch(share=False, inbrowser=True, server_name='127.0.0.1', server_port=6006, favicon_path="./favicon.ico")
