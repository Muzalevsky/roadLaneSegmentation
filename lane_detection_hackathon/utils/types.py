class Dict(dict):
    def keys(self) -> list:
        return list(super().keys())

    def values(self) -> list:
        return list(super().values())
