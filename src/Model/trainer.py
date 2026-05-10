from lightning import pytorch as pl

def start_training(model=None,
                   train_loader=None,
                   val_loader=None,
                   epoches=10
                   ):
        
        trainer = pl.Trainer(
                enable_checkpointing=False,
                logger=False,
                enable_progress_bar=True,
                accelerator="auto",
                devices=1,
                max_epochs=epoches
                )
        trainer.fit(model,train_loader, val_loader)
        return trainer