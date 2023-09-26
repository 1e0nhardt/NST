import gradio as gr
from pipeline_main import GatysPipeline, GatysConfig

def pyramind_style_transfer(
    content, style, strategy, resize, max_iter, loss,
    pyr_levels, opt_levels
    ):
    args = GatysConfig(is_demo=True)
    args.exact_resize = resize
    args.optimize_strategy = strategy
    args.max_iter = max_iter
    args.method = loss
    args.pyramid_levels = int(pyr_levels)
    args.max_scales = int(opt_levels)
    pipeline = GatysPipeline(args)
    return pipeline(content, style)

def change_strategy(choice):
    if choice == 'pyramid':
        return [gr.update(visible=True), gr.update(visible=True), gr.update(choices=['ours'], value='ours')]
    elif choice == 'common':
        return [gr.update(visible=False), gr.update(visible=False), gr.update(choices=["mean", "adain", 'efdm', 'cov', 'gatys', 'gatysdivc', 'l2'], value='cov')]

with gr.Blocks() as demo:
    gr.Markdown('Style Transfer with Image Pyramid')
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown('Content')
        with gr.Column(scale=1):
            gr.Markdown('Style')
    with gr.Row():
        content = gr.Image(type='filepath')
        style = gr.Image(type='filepath')
    with gr.Row():
        with gr.Column(scale=1, min_width=600):
            # with gr.Tab('pyramid'):
            strategy = gr.Dropdown(["common", "pyramid"], value='pyramid', label="Image Representation")
            loss = gr.Dropdown(['ours'], value='ours', label="Style Loss")
            exact_resize = gr.Checkbox(True, label="Resize to 512x512", interactive=True)
            max_iter = gr.Number(200, label='Max Iteration', info='Recommand: 200 for pyramind, 300 for Common.')
            pyr_levels = gr.Number(8, label='Levels of Pyramid', info='Decrypt image into # levels.')
            opt_levels = gr.Number(2, label='Levels of Optimize', info='Do style transfer on # levels in a cascade manner.')
            strategy.change(change_strategy, inputs=strategy, outputs=[pyr_levels, opt_levels, loss])
            btn = gr.Button("Go")
        with gr.Column(scale=2, min_width=600):
            output = gr.Image()

    btn.click(
        fn=pyramind_style_transfer, 
        inputs=[
            content, style, 
            strategy, exact_resize, max_iter, loss,
            pyr_levels, opt_levels], 
        outputs=output
    )

demo.launch(share=True)

